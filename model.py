# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix, lil_matrix


class DRDM(nn.Module):
    def __init__(self, config, dataset):
        super(DRDM, self).__init__()

        if len(dataset) == 5:
            train_manager, train_adj, disease_adj, drug_adj, pos_weight = dataset
            pretrain_embeddings = None
        elif len(dataset) == 6:
            train_manager, train_adj, disease_adj, drug_adj, pos_weight, pretrain_embeddings = dataset
        else:
            raise ValueError('DRDM dataset tuple must contain 5 or 6 elements.')

        self.interaction_matrix = coo_matrix(train_adj).astype(np.float32)
        self.disease_adj = disease_adj
        self.drug_adj = drug_adj
        indices = np.vstack((self.interaction_matrix.row, self.interaction_matrix.col)).astype(np.int64)
        values = self.interaction_matrix.data.astype(np.float32)
        indices_tensor = torch.from_numpy(indices)
        values_tensor = torch.from_numpy(values)
        self.dr_adj = torch.sparse_coo_tensor(
            indices_tensor,
            values_tensor,
            torch.Size(self.interaction_matrix.shape)
        ).coalesce()
        self.rd_adj = self.dr_adj.transpose(0, 1)
        self.path = './drug_data/' + str(config['dataset'])
        self.device = config['device']
        self.exp_coff = config['exp_coff']
        self.n_diseases = self.interaction_matrix.shape[0]
        self.n_drugs = self.interaction_matrix.shape[1]
        self.Graph = self.getSparseGraph()
        self.Graph2 = self.getSparseGraph2()
        self.dropout = nn.Dropout(p=float(config.get('dropout', 0.1)), inplace=False)
        self.pos_weight = pos_weight
        self.batch_size = config['batch_size']
        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.wd = config['wd']
        self.wr = config['wr']
        self.pretrain_freeze = bool(config.get('pretrain_freeze', False))
        self.pretrain_init_scale = float(config.get('pretrain_init_scale', 0.1))

        # Neighbor self-gating switch.
        # ablate_neighbor_self_gating=True makes the neighbor-graph branch use
        # the same initial embeddings as the association-graph branch, instead
        # of self-gated disease/drug embeddings.
        #
        # Removed modules:
        # - dual-view contrastive learning
        # - sign-based embedding noise
        # - topology-prior residual score term
        # - non-legacy propagation weight modes
        # - learnable decoder-scale gates
        self.ablate_neighbor_self_gating = bool(config.get('ablate_neighbor_self_gating', True))

        # Explicit Topology Decoder / TTM-D switches.
        # use_topology_decoder is the ablation-version name; use_ttm is kept as
        # a compatibility alias for arguments copied from drdm57x.
        self.use_topology_decoder = bool(config.get('use_topology_decoder', config.get('use_ttm', False)))
        self.topo_hidden = int(config.get('topo_hidden', 256))
        self.topo_alpha = float(config.get('topo_alpha', 0.2))
        self.rel_alpha = float(config.get('rel_alpha', 0.3))
        self.topo_loss_weight = float(config.get('topo_loss_weight', 0.05))
        self.degree_tau = float(config.get('degree_tau', 5.0))

        # Fixed settings after removing unused modules.
        self.prior_mode = 'none'
        self.prop_weight_mode = 'legacy'

        self.disease_embedding, self.drug_embedding = self._init_node_embeddings(
            pretrain_embeddings=pretrain_embeddings,
            use_pretrain=bool(config.get('use_pretrain_embed', True)),
        )

        # self-gating
        self.gating_weightdb = nn.Parameter(torch.FloatTensor(1, self.latent_dim))
        nn.init.xavier_normal_(self.gating_weightdb.data)
        self.gating_weightd = nn.Parameter(torch.FloatTensor(self.latent_dim, self.latent_dim))
        nn.init.xavier_normal_(self.gating_weightd.data)
        self.gating_weightrb = nn.Parameter(torch.FloatTensor(1, self.latent_dim))
        nn.init.xavier_normal_(self.gating_weightrb.data)
        self.gating_weightr = nn.Parameter(torch.FloatTensor(self.latent_dim, self.latent_dim))
        nn.init.xavier_normal_(self.gating_weightr.data)

        # Leakage-safe topology memory for the current fold.
        # It is constructed from train_adj, disease_adj and drug_adj only; hidden
        # validation/test associations never enter this prior.
        self._build_degree_buffers(train_adj)
        if self.use_topology_decoder:
            topo_prior = self._build_topology_prior(train_adj, disease_adj, drug_adj)
            self.register_buffer('topo_prior', topo_prior)
            self.register_buffer('topo_prior_mean', topo_prior.mean())

            self.topo_d_encoder = nn.Sequential(
                nn.Linear(self.n_drugs, self.topo_hidden),
                nn.ReLU(),
                nn.Dropout(p=float(config.get('dropout', 0.1))),
                nn.Linear(self.topo_hidden, self.latent_dim),
            )
            self.topo_r_encoder = nn.Sequential(
                nn.Linear(self.n_diseases, self.topo_hidden),
                nn.ReLU(),
                nn.Dropout(p=float(config.get('dropout', 0.1))),
                nn.Linear(self.topo_hidden, self.latent_dim),
            )

            # Inputs: disease embedding, drug embedding, elementwise product,
            # absolute difference, and topology prior scalar.
            self.rel_decoder = nn.Sequential(
                nn.Linear(self.latent_dim * 4 + 1, self.topo_hidden),
                nn.ReLU(),
                nn.Dropout(p=float(config.get('dropout', 0.1))),
                nn.Linear(self.topo_hidden, 1),
            )
        else:
            self.register_buffer(
                'topo_prior',
                torch.zeros((self.n_diseases, self.n_drugs), dtype=torch.float32, device=self.device)
            )
            self.register_buffer('topo_prior_mean', torch.tensor(0.0, dtype=torch.float32, device=self.device))

        print(
            'Removed modules: contrastive_learning, embedding_noise, topology_prior_residual, '
            'non_legacy_prop_weight_modes, learnable_decoder_scale. '
            'ablate_neighbor_self_gating={}, use_topology_decoder={}, topo_alpha={}, '
            'rel_alpha={}, prior_mode=none, topo_loss_weight={}, '
            'degree_tau={}, prop_weight_mode=legacy'.format(
                self.ablate_neighbor_self_gating,
                self.use_topology_decoder,
                self.topo_alpha,
                self.rel_alpha,
                self.topo_loss_weight,
                self.degree_tau,
            )
        )

    def _init_node_embeddings(self, pretrain_embeddings, use_pretrain=True):
        """Create disease/drug embedding layers.

        Original DRDM uses randomly initialized nn.Embedding layers. When
        drug_embedx/disease_embedx are provided, this method only replaces that
        initialization. Embeddings are trainable by default, preserving the
        original optimization behavior.
        """
        if (not use_pretrain) or pretrain_embeddings is None:
            disease_embedding = nn.Embedding(num_embeddings=self.n_diseases, embedding_dim=self.latent_dim)
            drug_embedding = nn.Embedding(num_embeddings=self.n_drugs, embedding_dim=self.latent_dim)
            print('Using original random node embeddings.')
            return disease_embedding, drug_embedding

        disease_raw = self._clean_pretrain_array(pretrain_embeddings['disease'], self.n_diseases, 'disease_embedx')
        drug_raw = self._clean_pretrain_array(pretrain_embeddings['drug'], self.n_drugs, 'drug_embedx')
        disease_init, drug_init = self._adapt_pretrain_pair(disease_raw, drug_raw, self.latent_dim)

        disease_weight = torch.from_numpy(disease_init.astype(np.float32))
        drug_weight = torch.from_numpy(drug_init.astype(np.float32))

        disease_embedding = nn.Embedding.from_pretrained(disease_weight, freeze=self.pretrain_freeze)
        drug_embedding = nn.Embedding.from_pretrained(drug_weight, freeze=self.pretrain_freeze)
        print(
            'Using pretrained node embeddings: '
            'disease {} -> {}, drug {} -> {}, freeze={}'.format(
                tuple(disease_raw.shape), tuple(disease_weight.shape),
                tuple(drug_raw.shape), tuple(drug_weight.shape), self.pretrain_freeze
            )
        )
        return disease_embedding, drug_embedding

    @staticmethod
    def _row_l2_normalize(x, eps=1e-12):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, eps)

    def _clean_pretrain_array(self, x, expected_rows, name):
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2:
            arr = arr.reshape(expected_rows, -1)
        if arr.shape[0] != expected_rows:
            raise ValueError('{} row count mismatch: got {}, expected {}.'.format(name, arr.shape[0], expected_rows))
        if arr.shape[1] <= 0:
            raise ValueError('{} has invalid shape: {}.'.format(name, arr.shape))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        arr = self._row_l2_normalize(arr).astype(np.float32)
        return arr

    @staticmethod
    def _standardize_by_columns(x):
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return ((x - mean) / std).astype(np.float32)

    @staticmethod
    def _reduce_or_pad_by_svd(x, target_dim):
        current_dim = x.shape[1]
        if current_dim == target_dim:
            return x.astype(np.float32)

        if current_dim > target_dim:
            # Deterministic PCA-like projection. This adds no learned/random layer.
            _, _, vt = np.linalg.svd(x, full_matrices=False)
            keep_dim = min(target_dim, vt.shape[0])
            reduced = np.matmul(x, vt[:keep_dim].T)
            if keep_dim < target_dim:
                pad = np.zeros((x.shape[0], target_dim - keep_dim), dtype=np.float32)
                reduced = np.concatenate([reduced, pad], axis=1)
            return reduced.astype(np.float32)

        pad = np.zeros((x.shape[0], target_dim - current_dim), dtype=np.float32)
        return np.concatenate([x, pad], axis=1).astype(np.float32)

    def _scale_embedding_init(self, x):
        x = self._row_l2_normalize(x).astype(np.float32)
        return (x * self.pretrain_init_scale).astype(np.float32)

    def _adapt_pretrain_pair(self, disease_raw, drug_raw, target_dim):
        if disease_raw.shape[1] == drug_raw.shape[1]:
            # Shared basis keeps disease and drug vectors in the same coordinate
            # space after reducing 768-dim semantic vectors to the DRDM latent dim.
            all_raw = np.concatenate([disease_raw, drug_raw], axis=0)
            all_z = self._standardize_by_columns(all_raw)
            all_latent = self._reduce_or_pad_by_svd(all_z, target_dim)
            all_latent = self._scale_embedding_init(all_latent)
            disease_init = all_latent[:self.n_diseases]
            drug_init = all_latent[self.n_diseases:]
            return disease_init, drug_init

        print(
            'Warning: disease_embedx and drug_embedx have different raw dimensions '
            '({} vs {}). Reducing them separately.'.format(disease_raw.shape[1], drug_raw.shape[1])
        )
        disease_z = self._standardize_by_columns(disease_raw)
        drug_z = self._standardize_by_columns(drug_raw)
        disease_init = self._scale_embedding_init(self._reduce_or_pad_by_svd(disease_z, target_dim))
        drug_init = self._scale_embedding_init(self._reduce_or_pad_by_svd(drug_z, target_dim))
        return disease_init, drug_init

    @staticmethod
    def _row_normalize_np(x):
        x = np.asarray(x, dtype=np.float32)
        row_sum = x.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        return x / row_sum

    def _build_topology_prior(self, train_adj, disease_adj, drug_adj):
        """Build the explicit topology prior from current-fold training data only.

        P_topo = 0.20 R + 0.30 Sd R + 0.30 R Sr + 0.20 Sd R Sr

        R is the current fold train_adj. Disease/drug similarity matrices are
        TopK graphs built from input similarities. The prior is normalized within
        this fold, so it does not leak hidden test edges.
        """
        R = np.asarray(train_adj, dtype=np.float32)
        Sd = self._row_normalize_np(disease_adj)
        Sr = self._row_normalize_np(drug_adj)

        P = (
            0.20 * R +
            0.30 * np.matmul(Sd, R) +
            0.30 * np.matmul(R, Sr) +
            0.20 * np.matmul(np.matmul(Sd, R), Sr)
        )

        max_v = float(P.max()) if P.size > 0 else 0.0
        if max_v > 0.0:
            P = P / max_v
        P = np.clip(P, 0.0, 1.0).astype(np.float32)
        return torch.tensor(P, dtype=torch.float32, device=self.device)

    def _build_degree_buffers(self, train_adj):
        """Degree buffers used by degree-aware topology injection, computed from train_adj only."""
        R = np.asarray(train_adj, dtype=np.float32)
        disease_deg = R.sum(axis=1)
        drug_deg = R.sum(axis=0)
        self.register_buffer('disease_deg', torch.tensor(disease_deg, dtype=torch.float32, device=self.device))
        self.register_buffer('drug_deg', torch.tensor(drug_deg, dtype=torch.float32, device=self.device))

    def self_gatingd(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weightd) + self.gating_weightdb))

    def self_gatingr(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weightr) + self.gating_weightrb))

    def get_weighted_adj_emb(self, adj, emb):
        """Graph propagation with fixed legacy message weighting and no embedding noise."""
        indices = adj._indices()
        values = adj._values()

        start_emb = emb[indices[0]].detach()
        end_emb = emb[indices[1]]
        cross_product = torch.mul(start_emb, end_emb).mean(dim=1)
        mat = (2 - 2 * cross_product) / self.exp_coff

        mat = mat * values
        new_indices = indices[0].unsqueeze(1).expand(end_emb.shape)
        weighted_emb = torch.mul(self.dropout(end_emb), mat.unsqueeze(1).expand(end_emb.shape))
        update_all_emb = torch.zeros(emb.shape, device=self.device)
        update_all_emb.scatter_add_(0, new_indices, weighted_emb)
        return update_all_emb

    def getSparseGraph(self):
        adj_mat = sp.dok_matrix((self.n_diseases + self.n_drugs, self.n_diseases + self.n_drugs), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_diseases, self.n_diseases:] = R
        adj_mat[self.n_diseases:, :self.n_diseases] = R.T
        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        rowsum[rowsum == 0.] = 1.
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().to(self.device)
        return Graph

    def getSparseGraph2(self):
        adj_mat = sp.dok_matrix((self.n_diseases + self.n_drugs, self.n_diseases + self.n_drugs), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R1 = lil_matrix(self.disease_adj)
        R2 = lil_matrix(self.drug_adj)
        adj_mat[:self.n_diseases, :self.n_diseases] = R1
        adj_mat[self.n_diseases:, self.n_diseases:] = R2
        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        rowsum[rowsum == 0.] = 1.
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().to(self.device)
        return Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def get_ego_embeddings(self):
        disease_embeddings = self.disease_embedding.weight
        drug_embeddings = self.drug_embedding.weight
        ego_embeddings = torch.cat([disease_embeddings, drug_embeddings], dim=0)

        if self.ablate_neighbor_self_gating:
            # Ablation: the neighbor graph branch starts from exactly the same
            # disease/drug embeddings as the association graph branch.
            ego_gating_embeddings = ego_embeddings
        else:
            dd_embeddings = self.self_gatingd(disease_embeddings)
            rr_embeddings = self.self_gatingr(drug_embeddings)
            ego_gating_embeddings = torch.cat([dd_embeddings, rr_embeddings], dim=0)

        return ego_embeddings, ego_gating_embeddings

    def _get_backbone_embeddings(self):
        """Two-branch DRDM encoder shared by original and topology decoders."""
        all_embeddings, all_gating_embeddings = self.get_ego_embeddings()
        d_r_embeddings_list = [all_embeddings]
        dd_rr_embeddings_list = [all_gating_embeddings]
        all_emb = all_embeddings
        all_emb_gating = all_gating_embeddings
        for _ in range(1, self.n_layers + 1):
            all_emb = self.get_weighted_adj_emb(self.Graph, all_emb)
            all_emb_gating = self.get_weighted_adj_emb(self.Graph2, all_emb_gating)
            d_r_embeddings_list.append(all_emb)
            dd_rr_embeddings_list.append(all_emb_gating)

        d_r_embeddings = torch.stack(d_r_embeddings_list, dim=1).mean(dim=1)
        dd_rr_embeddings = torch.stack(dd_rr_embeddings_list, dim=1).mean(dim=1)
        dr_diseaseEmbedding, dr_drugEmbedding = torch.split(d_r_embeddings, [self.n_diseases, self.n_drugs])
        diseaseEmbedding, drugEmbedding = torch.split(dd_rr_embeddings, [self.n_diseases, self.n_drugs])
        self.dr_diseaseEmbedding = dr_diseaseEmbedding
        self.dr_drugEmbedding = dr_drugEmbedding
        self.diseaseEmbedding = diseaseEmbedding
        self.drugEmbedding = drugEmbedding
        self.fuse_diseaseEmbedding = self.wd * self.dr_diseaseEmbedding + (1 - self.wd) * self.diseaseEmbedding
        self.fuse_drugEmbedding = self.wr * self.dr_drugEmbedding + (1 - self.wr) * self.drugEmbedding
        return self.fuse_diseaseEmbedding, self.fuse_drugEmbedding

    def _degree_gate(self, diseases, drugs):
        d_gate = torch.exp(-self.disease_deg[diseases] / self.degree_tau)
        r_gate = torch.exp(-self.drug_deg[drugs] / self.degree_tau)
        return d_gate, r_gate

    def forward(self, batch, is_training):
        diseases, drugs, labels = batch
        diseases = torch.LongTensor(diseases).to(self.device)
        drugs = torch.LongTensor(drugs).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)

        fuse_diseaseEmbedding, fuse_drugEmbedding = self._get_backbone_embeddings()
        batch_disease_all_embeddings = fuse_diseaseEmbedding[diseases]
        batch_drug_all_embeddings = fuse_drugEmbedding[drugs]

        d_gate, r_gate = self._degree_gate(diseases, drugs)
        topo_loss = torch.tensor(0.0, device=self.device)

        if self.use_topology_decoder:
            # Explicit topology decoding copied from drdm57x / TTM-D.
            topo_d_all = self.topo_d_encoder(self.topo_prior)
            topo_r_all = self.topo_r_encoder(self.topo_prior.t())
            topo_d = topo_d_all[diseases]
            topo_r = topo_r_all[drugs]

            # Low-degree nodes receive stronger topology-memory injection.
            batch_disease_all_embeddings = batch_disease_all_embeddings + self.topo_alpha * d_gate.unsqueeze(1) * topo_d
            batch_drug_all_embeddings = batch_drug_all_embeddings + self.topo_alpha * r_gate.unsqueeze(1) * topo_r

            dot_logit = torch.mul(batch_disease_all_embeddings, batch_drug_all_embeddings).sum(dim=1)
            prior = self.topo_prior[diseases, drugs].clamp(0.0, 1.0)

            rel_input = torch.cat([
                batch_disease_all_embeddings,
                batch_drug_all_embeddings,
                batch_disease_all_embeddings * batch_drug_all_embeddings,
                torch.abs(batch_disease_all_embeddings - batch_drug_all_embeddings),
                prior.unsqueeze(1),
            ], dim=1)
            rel_logit = self.rel_decoder(rel_input).squeeze(1)
            scores = dot_logit + self.rel_alpha * rel_logit

            topo_logit = torch.mul(topo_d, topo_r).sum(dim=1)
            base_weight = self.pos_weight * labels + 1.0 - labels
            topo_loss = F.binary_cross_entropy_with_logits(topo_logit, labels, weight=base_weight)
            BCE_loss = self.bce_logits_loss_fn(scores, labels)
        else:
            scores = torch.mul(batch_disease_all_embeddings, batch_drug_all_embeddings).sum(dim=1)
            # Preserve the original ablation loss path when topology decoder is disabled.
            BCE_loss = self.bce_loss_fn(torch.sigmoid(scores), labels)

        # Contrastive learning has been removed; optimize BCE (+ optional topology loss) only.
        ssl_loss = torch.tensor(0.0, device=self.device)
        loss = BCE_loss

        if self.use_topology_decoder:
            loss = loss + self.topo_loss_weight * topo_loss

        # Keep these for optional debugging/inspection without changing the return API.
        self.last_bce_loss = BCE_loss.detach()
        self.last_ssl_loss = ssl_loss.detach()
        self.last_topo_loss = topo_loss.detach()
        return loss, torch.sigmoid(scores)

    def predict(self, batch):
        _, _, labels = batch
        _, scores = self.forward(batch, False)
        return scores, labels

    def bce_logits_loss_fn(self, logits, label):
        logits = logits.reshape(-1)
        label = label.reshape(-1)
        weight = self.pos_weight * label + 1.0 - label
        loss = F.binary_cross_entropy_with_logits(input=logits, target=label, weight=weight)
        return loss

    def bce_loss_fn(self, predict, label):
        predict = predict.reshape(-1)
        label = label.reshape(-1)
        weight = self.pos_weight * label + 1 - label
        loss = F.binary_cross_entropy(input=predict, target=label, weight=weight)
        return loss
