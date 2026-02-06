import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import csv
import numpy as np
import scipy.sparse as sp
from time import time

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, x, y):
        super(Load_Dataset, self).__init__()

        # self.dataset = dataset

        self.miRNA = x[:,0]
        self.circRNA = x[:,1]
        self.y = y
        # self.len = self.x.shape[0]

    def __getitem__(self, index):

        return self.miRNA[index], self.circRNA[index], self.y[index]

    def __len__(self):
        return len(self.miRNA)


def read_csv(save_list, file_name):
    csv_reader = csv.reader(open(file_name))
    for row in csv_reader:
        miR, circ = row
        save_list.append([int(miR),int(circ)])
    return


# def get_adj_mat(n_users, n_items, R, dataset_path):
#     try:
#         t1 = time()
#         adj_mat = sp.load_npz(f'{dataset_path}/s_adj_mat.npz')
#         norm_adj_mat = sp.load_npz(f'{dataset_path}/s_norm_adj_mat.npz')
#         # mean_adj_mat = sp.load_npz('s_mean_adj_mat.npz')
#         print('already load adj matrix', adj_mat.shape, time() - t1)
#
#     except Exception:
#         adj_mat, norm_adj_mat = create_adj_mat(n_users, n_items, R)
#         sp.save_npz(f'{dataset_path}/s_adj_mat.npz', adj_mat)
#         sp.save_npz(f'{dataset_path}/s_norm_adj_mat.npz', norm_adj_mat)
#         # sp.save_npz('s_mean_adj_mat.npz', mean_adj_mat)
#     return adj_mat, norm_adj_mat

def create_adj_mat(n_users, n_items, R):
    t1 = time()
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()

    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape, time() - t1)

    t2 = time()

    def mean_adj_single(adj):
        # D^-1 * A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    def normalized_adj_single(adj):
        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    # norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))

    print('already normalize adjacency matrix', time() - t2)
    return adj_mat.tocsr(), norm_adj_mat.tocsr()


def data_generator(pos_train_file, neg_train_file, pos_test_file, neg_test_file, args):
    # try:
    #     train_dataset = torch.load("./data/{}_train.pt".format(protein))#train ../CircSSNN
    #     test_dataset = torch.load("./data/{}_test.pt".format(protein)) #test
    # except:
    #     print("{}数据集加载出错，重新生成！".format(protein))
    # train_dataset, test_dataset = get_data(protein)
    batch_size = args.batch_size
    # dataset_path = f'Dataset/{args.dataset}'

    train_data_pos, test_data_pos = [], []
    train_data_neg, test_data_neg = [], []

    # train_y, test_y = [], []
    read_csv(train_data_pos, pos_train_file)
    train_y = [1] * len(train_data_pos)

    read_csv(train_data_neg, neg_train_file)
    train_y += [0] * (len(train_data_neg))

    read_csv(test_data_pos, pos_test_file)
    test_y = [1] * len(test_data_pos)

    read_csv(test_data_neg, neg_test_file)
    test_y += [0] * (len(test_data_neg))

    n_users, n_items = 0, 0
    for elem in train_data_pos+test_data_pos: # 统计最大节点index
        n_users = max(n_users, elem[0])
        n_items = max(n_items, elem[1])

    n_users += 1
    n_items += 1

    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    for elem in train_data_pos:
        R[elem[0], elem[1]] = 1.

    plain_adj, norm_adj = create_adj_mat(n_users, n_items, R)

    train_dataset = Load_Dataset(np.array(train_data_pos+train_data_neg), np.array(train_y, dtype=float))
    test_dataset = Load_Dataset(np.array(test_data_pos+test_data_neg), np.array(test_y, dtype=float))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=False, # 去掉末尾不够batch_size的样本configs.drop_last
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=True, drop_last=False, # 去掉末尾不够batch_size的样本configs.drop_last
                                              num_workers=0)
    # train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, drop_last=True)

    return train_loader, test_loader, norm_adj, n_users, n_items
