import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from CTGM import DGCLCMIWithGatedMechanism
from dataloader import data_generator
import warnings
from time import time
import numpy as np
import csv
from sklearn.metrics import confusion_matrix, precision_score, recall_score, matthews_corrcoef, accuracy_score, \
    roc_auc_score, average_precision_score
import os
from pathlib import Path
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="DGCL-CMI: 基于深度学习的circRNA-miRNA相互作用预测")
    parser.add_argument('--dataset', nargs='?', default='CMI-9589',
                        help='Choose a dataset from {CMI-9589, CMI-9905, CMI-20208}')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[128,128,128]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--node_dropout', nargs='?', default='[0.3]',
                        help='Keep probability w.r.t. node dropout')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.3,0.3,0.3]',
                        help='Keep probability w.r.t. message dropout')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Negative slope of Leaky ReLU.')
    parser.add_argument('--l2_reg', type=float, default=5e-5,
                        help='L2 regularization coefficient.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing for BCE loss')
    parser.add_argument('--transformer_dropout', type=float, default=0.1,
                        help='Dropout rate for transformer layers')
    return parser.parse_args()

def ReadMyCsv1(SaveList, fileName):
    try:
        with open(fileName, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if row:
                    SaveList.append(row)
    except Exception as e:
        print(f"读取CSV文件 {fileName} 时出错: {e}")

def ReadMyCsv3(SaveList, fileName):
    try:
        with open(fileName, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if row:
                    counter = 1
                    while counter < len(row):
                        try:
                            row[counter] = float(row[counter])
                        except ValueError:
                            row[counter] = 0.0
                        counter += 1
                    SaveList.append(row)
    except Exception as e:
        print(f"读取CSV文件 {fileName} 时出错: {e}")

def GenerateEmbeddingFeature(SequenceList, EmbeddingList, PaddingLength):
    SampleFeature = []
    counter = 0
    while counter < len(SequenceList):
        PairFeature = []
        FeatureMatrix = []
        counter1 = 0
        while counter1 < PaddingLength:
            row = [0] * (len(EmbeddingList[0]) - 1)
            FeatureMatrix.append(row)
            counter1 += 1
        try:
            counter3 = 0
            while counter3 < PaddingLength and counter3 < len(SequenceList[counter][1]):
                counter4 = 0
                while counter4 < len(EmbeddingList):
                    if SequenceList[counter][1][counter3] == EmbeddingList[counter4][0]:
                        emb = np.array(EmbeddingList[counter4][1:], dtype=np.float32)
                        FeatureMatrix[counter3] = emb.tolist()
                        break
                    counter4 += 1
                counter3 += 1
        except Exception as e:
            print(f"生成嵌入特征时出错: {e}")
        PairFeature.append(FeatureMatrix)
        SampleFeature.append(PairFeature)
        counter += 1
    return SampleFeature

def count_csv_samples(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"统计文件 {file_path} 样本数时出错: {e}")
        return -1

def model_train(model, optimizer, criterion, train_loader, args):
    model.train()
    total_loss = []
    total_metrics = {
        'specificity': [], 'precision': [], 'sensitivity': [],
        'mcc': [], 'accuracy': [], 'roc_auc': [], 'aupr': []
    }
    for batch_idx, (miRNA, circRNA, y_true) in enumerate(train_loader):
        miRNA = miRNA.long().to(args.device)
        circRNA = circRNA.long().to(args.device)
        y_true = y_true.to(args.device)
        u_embeddings, i_embeddings = model(miRNA, circRNA, drop_flag=True)
        y_scores = torch.mm(u_embeddings, i_embeddings.T).diag()
        loss = criterion(y_scores, y_true)
        l2_loss = 0.0
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                l2_loss += torch.norm(param, 2)
        loss += args.l2_reg * l2_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
        optimizer.step()
        y_scores_np = y_scores.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        y_pred = np.where(y_scores_np >= 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision = precision_score(y_true_np, y_pred) if (tp + fp) != 0 else 0
        sensitivity = recall_score(y_true_np, y_pred) if (tp + fn) != 0 else 0
        mcc = matthews_corrcoef(y_true_np, y_pred)
        accuracy = accuracy_score(y_true_np, y_pred)
        roc_auc = roc_auc_score(y_true_np, y_scores_np)
        aupr = average_precision_score(y_true_np, y_scores_np)
        total_loss.append(loss.item())
        total_metrics['specificity'].append(specificity)
        total_metrics['precision'].append(precision)
        total_metrics['sensitivity'].append(sensitivity)
        total_metrics['mcc'].append(mcc)
        total_metrics['accuracy'].append(accuracy)
        total_metrics['roc_auc'].append(roc_auc)
        total_metrics['aupr'].append(aupr)
        if (batch_idx + 1) % 5 == 0:
            print(f"Train Batch {batch_idx + 1}: "
                  f"Loss: {loss.item():.4f} | "
                  f"AUC: {roc_auc:.4f} | "
                  f"Acc: {accuracy:.4f} | "
                  f"MCC: {mcc:.4f}")
    avg_loss = np.mean(total_loss)
    avg_metrics = {k: np.mean(v) for k, v in total_metrics.items()}
    return avg_loss, avg_metrics

def model_test(model, criterion, test_loader, args):
    model.eval()
    total_loss = []
    total_metrics = {
        'specificity': [], 'precision': [], 'sensitivity': [],
        'mcc': [], 'accuracy': [], 'roc_auc': [], 'aupr': []
    }
    with torch.no_grad():
        for batch_idx, (miRNA, circRNA, y_true) in enumerate(test_loader):
            miRNA = miRNA.long().to(args.device)
            circRNA = circRNA.long().to(args.device)
            y_true = y_true.to(args.device)
            u_embeddings, i_embeddings = model(miRNA, circRNA)
            y_scores = torch.mm(u_embeddings, i_embeddings.T).diag()
            loss = criterion(y_scores, y_true)
            total_loss.append(loss.item())
            y_scores_np = y_scores.detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()
            y_pred = np.where(y_scores_np >= 0.5, 1, 0)
            tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            precision = precision_score(y_true_np, y_pred) if (tp + fp) != 0 else 0
            sensitivity = recall_score(y_true_np, y_pred) if (tp + fn) != 0 else 0
            mcc = matthews_corrcoef(y_true_np, y_pred)
            accuracy = accuracy_score(y_true_np, y_pred)
            roc_auc = roc_auc_score(y_true_np, y_scores_np)
            aupr = average_precision_score(y_true_np, y_scores_np)
            total_metrics['specificity'].append(specificity)
            total_metrics['precision'].append(precision)
            total_metrics['sensitivity'].append(sensitivity)
            total_metrics['mcc'].append(mcc)
            total_metrics['accuracy'].append(accuracy)
            total_metrics['roc_auc'].append(roc_auc)
            total_metrics['aupr'].append(aupr)
    avg_loss = np.mean(total_loss)
    avg_metrics = {k: np.mean(v) for k, v in total_metrics.items()}
    avg_metrics['loss'] = avg_loss
    return avg_metrics

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    warnings.filterwarnings('ignore')
    args = parse_args()
    args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    print(f"使用设备: {args.device}")
    print(f"参数设置: {args}")
    print(f"消息传递层数: {len(args.mess_dropout)}层")
    for data_name in ['CMI-9589']:
        args.dataset = data_name
        results_dir = f'result/{args.dataset}'
        os.makedirs(results_dir, exist_ok=True)
        for fold in [0]:
            print(f"\n{'=' * 50}")
            print(f"数据集: {data_name}, 折数: {fold}")
            print(f"{'=' * 50}")
            try:
                pos_train_path = f'Dataset/{args.dataset}/Positive_Sample_Train{fold}.csv'
                neg_train_path = f'Dataset/{args.dataset}/Negative_Sample_Train{fold}.csv'
                pos_test_path = f'Dataset/{args.dataset}/Positive_Sample_Test{fold}.csv'
                neg_test_path = f'Dataset/{args.dataset}/Negative_Sample_Test{fold}.csv'
                pos_train_samples = count_csv_samples(pos_train_path)
                neg_train_samples = count_csv_samples(neg_train_path)
                pos_test_samples = count_csv_samples(pos_test_path)
                neg_test_samples = count_csv_samples(neg_test_path)
                print("\n【原始数据集样本数统计】")
                print(f"正训练集: {pos_train_samples} 条")
                print(f"负训练集: {neg_train_samples} 条")
                print(f"正测试集: {pos_test_samples} 条")
                print(f"负测试集: {neg_test_samples} 条")
                train_loader, test_loader, norm_adj, n_users, n_items = data_generator(
                    pos_train_path,
                    neg_train_path,
                    pos_test_path,
                    neg_test_path,
                    args)
                train_pos_count = sum(1 for _, _, y in train_loader for label in y if label == 1)
                train_neg_count = sum(1 for _, _, y in train_loader for label in y if label == 0)
                test_pos_count = sum(1 for _, _, y in test_loader for label in y if label == 1)
                test_neg_count = sum(1 for _, _, y in test_loader for label in y if label == 0)
                print("\n【加载后数据集样本数统计】")
                print(f"训练集 - 正样本: {train_pos_count}, 负样本: {train_neg_count}, 总计: {train_pos_count + train_neg_count}")
                print(f"测试集 - 正样本: {test_pos_count}, 负样本: {test_neg_count}, 总计: {test_pos_count + test_neg_count}")
                print(f"\nmiRNA数量: {n_users}, circRNA数量: {n_items}")
                embedding_file = f"Dataset/{args.dataset}/EmbeddingFeature.npz"
                if not os.path.exists(embedding_file):
                    raise FileNotFoundError(f"嵌入特征文件不存在: {embedding_file}")
                data = np.load(embedding_file)
                CircEmbeddingFeature = data['circRNA']
                miRNAEmbeddingFeature = data['miRNA']
                print(f"circRNA嵌入形状: {CircEmbeddingFeature.shape}")
                print(f"miRNA嵌入形状: {miRNAEmbeddingFeature.shape}")
                CircEmbeddingFeature = torch.tensor(CircEmbeddingFeature, dtype=torch.float).to(args.device)
                miRNAEmbeddingFeature = torch.tensor(miRNAEmbeddingFeature, dtype=torch.float).to(args.device)
                circ_struct_path = f"Dataset/{args.dataset}/circ_cgr.csv"
                mirna_struct_path = f"Dataset/{args.dataset}/mi_cgr.csv"
                model = DGCLCMIWithGatedMechanism(
                    n_users,
                    n_items,
                    norm_adj,
                    circ_struct_path,
                    mirna_struct_path,
                    args
                ).to(args.device)
                model.CircEmbeddingFeature = CircEmbeddingFeature
                model.miRNAEmbeddingFeature = miRNAEmbeddingFeature
                optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
                criterion = nn.CrossEntropyLoss()
                best_metrics = {
                    'epoch': 0, 'loss': float('inf'), 'roc_auc': 0.0, 'accuracy': 0.0,
                    'mcc': -1.0, 'sensitivity': 0.0, 'precision': 0.0, 'specificity': 0.0, 'aupr': 0.0
                }
                train_losses = []
                test_losses = []
                no_improve_epochs = 0
                best_model_path = f'{results_dir}/best_model_fold{fold}.pth'
                for epoch in range(args.epoch):
                    epoch_start_time = time()
                    train_loss, train_metrics = model_train(model, optimizer, criterion, train_loader, args)
                    train_losses.append(train_loss)
                    test_metrics = model_test(model, criterion, test_loader, args)
                    test_losses.append(test_metrics['loss'])
                    epoch_time = time() - epoch_start_time
                    print(f"\nEpoch {epoch + 1}/{args.epoch} | 时间: {epoch_time:.2f}秒")
                    print(f"训练: 损失={train_loss:.4f} | AUC={train_metrics['roc_auc']:.4f} | Acc={train_metrics['accuracy']:.4f}")
                    print(f"测试: 损失={test_metrics['loss']:.4f} | AUC={test_metrics['roc_auc']:.4f} | Acc={test_metrics['accuracy']:.4f} | MCC={test_metrics['mcc']:.4f}")
                    scheduler.step(test_metrics['roc_auc'])
                    if test_metrics['roc_auc'] > best_metrics['roc_auc'] + 1e-5:
                        best_metrics.update({
                            'epoch': epoch,
                            'loss': test_metrics['loss'],
                            'roc_auc': test_metrics['roc_auc'],
                            'accuracy': test_metrics['accuracy'],
                            'mcc': test_metrics['mcc'],
                            'sensitivity': test_metrics['sensitivity'],
                            'precision': test_metrics['precision'],
                            'specificity': test_metrics['specificity'],
                            'aupr': test_metrics['aupr']
                        })
                        torch.save(model.state_dict(), best_model_path)
                        print(f"更新最优模型 (AUC提升: {test_metrics['roc_auc'] - best_metrics['roc_auc'] + 1e-5:.6f})")
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                        print(f"AUC无提升，连续无提升轮次: {no_improve_epochs}/{args.patience}")
                    if no_improve_epochs >= args.patience:
                        print(f"连续{args.patience}个epoch AUC无提升，触发早停")
                        break
                with open(f"{results_dir}/result.txt", 'a', encoding='utf-8') as f:
                    f.write(f"数据集: {data_name}, 折数: {fold}\n")
                    f.write(f"原始训练集: 正样本{pos_train_samples}, 负样本{neg_train_samples}\n")
                    f.write(f"加载后训练集: 正样本{train_pos_count}, 负样本{train_neg_count}\n")
                    f.write(f"最优测试指标 - Epoch: {best_metrics['epoch'] + 1}\n")
                    f.write(f"损失: {best_metrics['loss']:.4f} | AUC: {best_metrics['roc_auc']:.4f} | "
                            f"Acc: {best_metrics['accuracy']:.4f} | MCC: {best_metrics['mcc']:.4f}\n")
                    f.write(f"灵敏度: {best_metrics['sensitivity']:.4f} | 特异度: {best_metrics['specificity']:.4f} | "
                            f"精确率: {best_metrics['precision']:.4f} | AUPR: {best_metrics['aupr']:.4f}\n")
                    f.write("-" * 50 + "\n")
                print(f"\n折数 {fold} 训练完成，结果已保存至 {results_dir}/result.txt")
            except Exception as e:
                print(f"折数 {fold} 训练出错: {e}")
                continue

if __name__ == "__main__":
    main()