import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attn_scores = torch.matmul(x, self.attention_weights).squeeze(-1)
        attn_weights = self.softmax(attn_scores)
        weighted_sum = torch.matmul(attn_weights.unsqueeze(1), x).squeeze(1)
        return weighted_sum


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device) *
            (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(seq_len, d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + pe
        return self.dropout(x)


class LightweightTransformer(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4, output_size=128, nhead=4, dropout=0.1):
        super(LightweightTransformer, self).__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.attention_pool = AttentionPooling(hidden_size)
        self.fc_mid = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.proj(x)
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)
        x = self.attention_pool(transformer_out)
        x = self.fc_mid(x)
        x = F.relu(x)
        x = self.layer_norm(x)
        x = self.fc_out(x)
        return self.output_norm(x)


class SimplifiedFeatureFusion(nn.Module):
    def __init__(self, transformer_dim, struct_dim, output_dim):
        super(SimplifiedFeatureFusion, self).__init__()
        self.transformer_proj = nn.Sequential(
            nn.Linear(transformer_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.t_weight = nn.Parameter(torch.tensor(0.5))
        self.s_weight = nn.Parameter(torch.tensor(0.5))
        self.fusion_net = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, transformer_feat, struct_feat):
        t_feat = self.transformer_proj(transformer_feat)
        s_feat = self.struct_proj(struct_feat)
        t_weight = torch.sigmoid(self.t_weight)
        s_weight = torch.sigmoid(self.s_weight)
        weighted_t = t_weight * t_feat
        weighted_s = s_weight * s_feat
        fused = weighted_t + weighted_s
        fused = self.fusion_net(fused)
        return fused


class DGCLCMIWithGatedMechanism(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, circ_struct_path, mirna_struct_path, args):
        super(DGCLCMIWithGatedMechanism, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.emb_size = args.embed_size
        self.norm_adj = norm_adj
        self.layers = eval(args.layer_size)
        self.alpha = args.alpha
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.device = args.device
        self.CircEmbeddingFeature = None
        self.miRNAEmbeddingFeature = None
        self.circ_struct_features = self._load_struct_features(circ_struct_path, is_circ=True)
        self.mirna_struct_features = self._load_struct_features(mirna_struct_path, is_circ=False)
        self.struct_feature_dim = self.circ_struct_features.shape[1]
        self._init_transformers()
        self.circ_fusion = SimplifiedFeatureFusion(self.emb_size, self.struct_feature_dim, self.emb_size)
        self.mirna_fusion = SimplifiedFeatureFusion(self.emb_size, self.struct_feature_dim, self.emb_size)
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(len(self.layers))
        ])
        self.weight_dict = self.init_type_aware_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.layers[i]) for i in range(len(self.layers))])

    def _init_transformers(self):
        if self.CircEmbeddingFeature is not None and self.miRNAEmbeddingFeature is not None:
            circ_input_dim = self.CircEmbeddingFeature.shape[2]
            mirna_input_dim = self.miRNAEmbeddingFeature.shape[2]
            self.circ_transformer = LightweightTransformer(
                input_size=circ_input_dim,
                hidden_size=512,
                num_layers=2,
                output_size=self.emb_size,
                nhead=4,
                dropout=0.1
            ).to(self.device)
            self.mirna_transformer = LightweightTransformer(
                input_size=mirna_input_dim,
                hidden_size=512,
                num_layers=2,
                output_size=self.emb_size,
                nhead=4,
                dropout=0.1
            ).to(self.device)
        else:
            self.circ_transformer = None
            self.mirna_transformer = None

    def _load_struct_features(self, file_path, is_circ):
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"结构特征文件不存在: {file_path}")
            df = pd.read_csv(
                file_path,
                header=None,
                dtype=np.float32
            )
            features = df.values
            num_samples = self.n_item if is_circ else self.n_user
            if features.shape[0] != num_samples:
                raise ValueError(f"结构特征数量不匹配，预期{num_samples}，实际{features.shape[0]}")
            return torch.FloatTensor(features).to(self.device)
        except Exception as e:
            print(f"加载结构特征时出错: {e}")
            raise RuntimeError("结构特征加载失败，程序终止")

    def init_type_aware_weight(self):
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            in_dim = layers[k]
            out_dim = layers[k + 1]
            weight_dict.update({f'W_c_{k}': nn.Parameter(initializer(torch.empty(in_dim, out_dim)))})
            weight_dict.update({f'b_c_{k}': nn.Parameter(torch.zeros(1, out_dim))})
            weight_dict.update({f'W_c_to_m_{k}': nn.Parameter(initializer(torch.empty(in_dim, out_dim)))})
            weight_dict.update({f'b_c_to_m_{k}': nn.Parameter(torch.zeros(1, out_dim))})
            weight_dict.update({f'W_m_{k}': nn.Parameter(initializer(torch.empty(in_dim, out_dim)))})
            weight_dict.update({f'b_m_{k}': nn.Parameter(torch.zeros(1, out_dim))})
            weight_dict.update({f'W_m_to_c_{k}': nn.Parameter(initializer(torch.empty(in_dim, out_dim)))})
            weight_dict.update({f'b_m_to_c_{k}': nn.Parameter(torch.zeros(1, out_dim))})
            weight_dict.update({f'W_f_c_{k}': nn.Parameter(initializer(torch.empty(2 * in_dim, 1)))})
            weight_dict.update({f'b_f_c_{k}': nn.Parameter(torch.tensor(0.5))})
            weight_dict.update({f'W_f_m_{k}': nn.Parameter(initializer(torch.empty(2 * in_dim, 1)))})
            weight_dict.update({f'b_f_m_{k}': nn.Parameter(torch.tensor(0.5))})
            weight_dict.update({f'W_i_c_{k}': nn.Parameter(initializer(torch.empty(2 * out_dim, 1)))})
            weight_dict.update({f'b_i_c_{k}': nn.Parameter(torch.zeros(1))})
            weight_dict.update({f'W_i_m_{k}': nn.Parameter(initializer(torch.empty(2 * out_dim, 1)))})
            weight_dict.update({f'b_i_m_{k}': nn.Parameter(torch.zeros(1))})
            weight_dict.update({f'W_ec_c_{k}': nn.Parameter(initializer(torch.empty(out_dim, out_dim)))})
            weight_dict.update({f'b_ec_c_{k}': nn.Parameter(torch.zeros(1, out_dim))})
            weight_dict.update({f'W_ec_m_{k}': nn.Parameter(initializer(torch.empty(out_dim, out_dim)))})
            weight_dict.update({f'b_ec_m_{k}': nn.Parameter(torch.zeros(1, out_dim))})
            weight_dict.update({f'W_o_c_{k}': nn.Parameter(initializer(torch.empty(out_dim, 1)))})
            weight_dict.update({f'b_o_c_{k}': nn.Parameter(torch.zeros(1))})
            weight_dict.update({f'W_o_m_{k}': nn.Parameter(initializer(torch.empty(out_dim, 1)))})
            weight_dict.update({f'b_o_m_{k}': nn.Parameter(torch.zeros(1))})
        return weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        v = v.masked_select(dropout_mask)
        i = i[:, dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, users, items, drop_flag=False, valid_auc_list=None):
        if self.CircEmbeddingFeature is None or self.miRNAEmbeddingFeature is None:
            raise ValueError("序列嵌入特征未初始化")
        if self.circ_transformer is None or self.mirna_transformer is None:
            self._init_transformers()
        if drop_flag and self.node_dropout > 0:
            A_hat = self.sparse_dropout(self.sparse_norm_adj, self.node_dropout,
                                        self.sparse_norm_adj._nnz())
        else:
            A_hat = self.sparse_norm_adj
        selected_circ_embeddings = self.CircEmbeddingFeature[items]
        selected_mirna_embeddings = self.miRNAEmbeddingFeature[users]
        circ_transformer_feat = self.circ_transformer(selected_circ_embeddings)
        mirna_transformer_feat = self.mirna_transformer(selected_mirna_embeddings)
        selected_circ_struct = self.circ_struct_features[items]
        selected_mirna_struct = self.mirna_struct_features[users]
        circ_emb = self.circ_fusion(circ_transformer_feat, selected_circ_struct)
        mirna_emb = self.mirna_fusion(mirna_transformer_feat, selected_mirna_struct)
        full_circ_emb = torch.zeros(self.n_item, self.emb_size, device=self.device)
        full_mirna_emb = torch.zeros(self.n_user, self.emb_size, device=self.device)
        full_circ_emb[items] = circ_emb
        full_mirna_emb[users] = mirna_emb
        ego_embeddings = torch.cat([full_mirna_emb, full_circ_emb], dim=0)
        all_embeddings = [ego_embeddings]
        optimal_layer = len(self.layers)
        for k in range(optimal_layer):
            miRNA_emb = ego_embeddings[:self.n_user, :]
            circ_emb = ego_embeddings[self.n_user:, :]
            indices = A_hat._indices()
            values = A_hat._values()
            mask = (indices[0] < self.n_user) & (indices[1] >= self.n_user)
            if mask.sum() == 0:
                circ_neighbor_miRNA_gated = 0.6 * circ_emb
            else:
                new_indices = torch.stack([indices[1][mask] - self.n_user, indices[0][mask]], dim=0)
                new_values = values[mask]
                new_shape = (self.n_item, self.n_user)
                edge_indices = new_indices.t()
                circ_indices = edge_indices[:, 0]
                miRNA_indices = edge_indices[:, 1]
                circ_features = circ_emb[circ_indices]
                miRNA_features = miRNA_emb[miRNA_indices]
                concat_features = torch.cat([circ_features, miRNA_features], dim=1)
                forget_weights = torch.sigmoid(
                    F.leaky_relu(torch.matmul(concat_features, self.weight_dict[f'W_f_c_{k}']), negative_slope=0.2) +
                    self.weight_dict[f'b_f_c_{k}']
                )
                new_values_gated = new_values * forget_weights.squeeze()
                gated_sparse_tensor = torch.sparse.FloatTensor(new_indices, new_values_gated, new_shape).to(
                    A_hat.device)
                circ_neighbor_miRNA_gated = torch.sparse.mm(gated_sparse_tensor, miRNA_emb)
            circ_neighbor_msg = self.layer_scales[k] * (
                    torch.matmul(circ_neighbor_miRNA_gated, self.weight_dict[f'W_c_to_m_{k}']) +
                    self.weight_dict[f'b_c_to_m_{k}']
            )
            circ_self_msg = self.layer_scales[k] * (
                    torch.matmul(circ_emb, self.weight_dict[f'W_c_{k}']) +
                    self.weight_dict[f'b_c_{k}']
            )
            circ_new_msg = circ_self_msg + circ_neighbor_msg
            circ_new_msg = self.layer_norms[k](circ_new_msg)
            W_i_c = self.weight_dict[f'W_i_c_{k}']
            in_dim = W_i_c.shape[0] // 2
            circ_input_gate = torch.sigmoid(
                0.8 * torch.matmul(circ_emb, W_i_c[:in_dim]) +
                0.2 * torch.matmul(circ_new_msg, W_i_c[in_dim:]) +
                self.weight_dict[f'b_i_c_{k}']
            )
            circ_candidate = torch.tanh(
                torch.matmul(circ_new_msg, self.weight_dict[f'W_ec_c_{k}']) +
                self.weight_dict[f'b_ec_c_{k}']
            )
            circ_emb_updated = (1 - circ_input_gate) * circ_emb + circ_input_gate * circ_candidate
            circ_emb_activated = F.leaky_relu(circ_emb_updated, negative_slope=self.alpha)
            circ_output_gate = torch.sigmoid(
                torch.matmul(circ_emb_activated, self.weight_dict[f'W_o_c_{k}']) +
                self.weight_dict[f'b_o_c_{k}']
            )
            circ_emb_final = circ_output_gate * torch.tanh(circ_emb_activated) * 1.2
            if mask.sum() == 0:
                miRNA_neighbor_circ_gated = 0.6 * miRNA_emb
            else:
                new_indices_mi = torch.stack([indices[0][mask], indices[1][mask] - self.n_user], dim=0)
                new_shape_mi = (self.n_user, self.n_item)
                edge_indices_mi = new_indices_mi.t()
                miRNA_indices_mi = edge_indices_mi[:, 0]
                circ_indices_mi = edge_indices_mi[:, 1]
                miRNA_features_mi = miRNA_emb[miRNA_indices_mi]
                circ_features_mi = circ_emb[circ_indices_mi]
                concat_features_mi = torch.cat([miRNA_features_mi, circ_features_mi], dim=1)
                forget_weights_mi = torch.sigmoid(
                    F.leaky_relu(torch.matmul(concat_features_mi, self.weight_dict[f'W_f_m_{k}']), negative_slope=0.2) +
                    self.weight_dict[f'b_f_m_{k}']
                )
                new_values_gated_mi = new_values * forget_weights_mi.squeeze()
                gated_sparse_tensor_mi = torch.sparse.FloatTensor(new_indices_mi, new_values_gated_mi, new_shape_mi).to(
                    A_hat.device)
                miRNA_neighbor_circ_gated = torch.sparse.mm(gated_sparse_tensor_mi, circ_emb)
            miRNA_neighbor_msg = self.layer_scales[k] * (
                    torch.matmul(miRNA_neighbor_circ_gated, self.weight_dict[f'W_m_to_c_{k}']) +
                    self.weight_dict[f'b_m_to_c_{k}']
            )
            miRNA_self_msg = self.layer_scales[k] * (
                    torch.matmul(miRNA_emb, self.weight_dict[f'W_m_{k}']) +
                    self.weight_dict[f'b_m_{k}']
            )
            miRNA_new_msg = miRNA_self_msg + miRNA_neighbor_msg
            miRNA_new_msg = self.layer_norms[k](miRNA_new_msg)
            W_i_m = self.weight_dict[f'W_i_m_{k}']
            in_dim_m = W_i_m.shape[0] // 2
            miRNA_input_gate = torch.sigmoid(
                0.8 * torch.matmul(miRNA_emb, W_i_m[:in_dim_m]) +
                0.2 * torch.matmul(miRNA_new_msg, W_i_m[in_dim_m:]) +
                self.weight_dict[f'b_i_m_{k}']
            )
            miRNA_candidate = torch.tanh(
                torch.matmul(miRNA_new_msg, self.weight_dict[f'W_ec_m_{k}']) +
                self.weight_dict[f'b_ec_m_{k}']
            )
            miRNA_emb_updated = (1 - miRNA_input_gate) * miRNA_emb + miRNA_input_gate * miRNA_candidate
            miRNA_emb_activated = F.leaky_relu(miRNA_emb_updated, negative_slope=self.alpha)
            miRNA_output_gate = torch.sigmoid(
                torch.matmul(miRNA_emb_activated, self.weight_dict[f'W_o_m_{k}']) +
                self.weight_dict[f'b_o_m_{k}']
            )
            miRNA_emb_final = miRNA_output_gate * torch.tanh(miRNA_emb_activated) * 1.2
            ego_embeddings = torch.cat([miRNA_emb_final, circ_emb_final], dim=0)
            all_embeddings.append(ego_embeddings)
        if drop_flag and self.mess_dropout[0] > 0:
            all_embeddings = [
                F.dropout(emb, p=self.mess_dropout[0], training=self.training)
                for emb in all_embeddings
            ]
        layer_wise_weights = F.softmax(torch.tensor([1.0 / (k + 1) for k in range(len(all_embeddings))]), dim=0).to(
            self.device)
        final_embeddings = torch.zeros_like(all_embeddings[0])
        for k in range(len(all_embeddings)):
            final_embeddings += layer_wise_weights[k] * all_embeddings[k]
        u_emb = final_embeddings[:self.n_user, :]
        i_emb = final_embeddings[self.n_user:, :]
        users_emb = u_emb[users, :]
        items_emb = i_emb[items, :]
        return users_emb, items_emb