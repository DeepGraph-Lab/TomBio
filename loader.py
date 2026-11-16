import math
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold

sys.path.append('.')


def config_model():
    config = OrderedDict()
    return config


def canonical_dataset_name(dataset):
    """Normalize dataset names used by the new MAT layout.

    The user clarified that Gdataset is the replacement for the old Fdataset.
    """
    dataset = str(dataset).strip()
    lower = dataset.lower()
    if lower == 'fdataset':
        return 'Gdataset'
    if lower == 'gdataset':
        return 'Gdataset'
    if lower == 'cdataset':
        return 'Cdataset'
    if lower == 'lrssl':
        return 'lrssl'
    raise ValueError("Unknown dataset '{}'. Expected one of: Gdataset, Fdataset, Cdataset, lrssl.".format(dataset))


def get_disease_sim_Matrix(disease_similarity, disease_disease_topk):
    disease_disease_Matrix = np.zeros((disease_similarity.shape[0], disease_similarity.shape[0]), np.float32)
    disease_sim_Matrix = disease_similarity
    for disease_num0 in range(disease_similarity.shape[0]):
        disease_sim = {}
        for disease_num1 in range(disease_similarity.shape[0]):
            if disease_num0 == disease_num1:
                continue
            disease_sim[disease_num1] = disease_sim_Matrix[disease_num0][disease_num1]
        sorted_disease_list = sorted(disease_sim.items(), key=lambda d: d[1], reverse=True)
        for i in range(min(disease_disease_topk, len(sorted_disease_list))):
            disease_num1 = sorted_disease_list[i][0]
            disease_disease_Matrix[disease_num0][disease_num1] = disease_sim_Matrix[disease_num0][disease_num1]

    return disease_disease_Matrix


def get_drug_sim_Matrix(drug_similarity, drug_drug_topk):
    drug_drug_Matrix = np.zeros((drug_similarity.shape[0], drug_similarity.shape[0]), np.float32)
    drug_sim_Matrix = drug_similarity
    for drug_num0 in range(drug_similarity.shape[0]):
        drug_sim = {}
        for drug_num1 in range(drug_similarity.shape[0]):
            if drug_num0 == drug_num1:
                continue
            drug_sim[drug_num1] = drug_sim_Matrix[drug_num0][drug_num1]
        sorted_drug_list = sorted(drug_sim.items(), key=lambda d: d[1], reverse=True)
        for i in range(min(drug_drug_topk, len(sorted_drug_list))):
            drug_num1 = sorted_drug_list[i][0]
            drug_drug_Matrix[drug_num0][drug_num1] = drug_sim_Matrix[drug_num0][drug_num1]

    return drug_drug_Matrix


def _extract_entity_names(mat_field, expected_len, prefer_column=0):
    """Return one name per entity from MATLAB cell/string arrays.

    Gdataset/Cdataset usually store Wdname/Wrname as n x 1 cells. LRSSL stores
    Wdname as n x 2 cells: column 0 is disease name and column 1 is MeSH ID.
    The model only needs the row count/order; names are retained for validation.
    """
    arr = np.asarray(mat_field)
    if arr.ndim == 2 and arr.shape[0] == expected_len:
        col = min(prefer_column, arr.shape[1] - 1)
        return arr[:, col].reshape(-1)
    flat = arr.reshape(-1)
    if flat.shape[0] < expected_len:
        raise ValueError('Name field has {} entries, expected at least {}.'.format(flat.shape[0], expected_len))
    return flat[:expected_len]


def _read_embedding_field(mat, primary_key, fallback_key, expected_rows, field_name):
    """Read a 2D embedding field from a MAT file with strict row alignment checks."""
    if primary_key in mat:
        emb = mat[primary_key]
        used_key = primary_key
    elif fallback_key in mat:
        emb = mat[fallback_key]
        used_key = fallback_key
    else:
        return None

    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        emb = emb.reshape(expected_rows, -1)
    if emb.shape[0] != expected_rows:
        raise ValueError(
            '{} row count mismatch from key {}: got {}, expected {}. '
            'Please make sure rows are aligned to Wrname/Wdname order.'.format(
                field_name, used_key, emb.shape[0], expected_rows
            )
        )
    if emb.shape[1] <= 0:
        raise ValueError('{} has invalid shape: {}.'.format(field_name, emb.shape))
    if not np.isfinite(emb).all():
        print('Warning: non-finite values found in {}; they will be replaced during model initialization.'.format(field_name))
    return emb.astype(np.float32)


def resolve_mat_path(dataset, mat_path=None, data_root='./drug_data'):
    """Resolve the MAT path for the user's new file layout.

    Defaults:
      Gdataset / old Fdataset -> ./drug_data/Gdataset/Gdataset.mat
      Cdataset                -> ./drug_data/Cdataset/Cdataset.mat
      lrssl                   -> ./drug_data/lrssl/lrssl.mat
    """
    if mat_path:
        if not os.path.exists(mat_path):
            raise FileNotFoundError('--mat_path does not exist: {}'.format(mat_path))
        return mat_path

    dataset = canonical_dataset_name(dataset)
    data_root = str(data_root)

    if dataset == 'Gdataset':
        dataset_dir = os.path.join(data_root, 'Gdataset')
        candidates = [
            os.path.join(dataset_dir, 'Gdataset.mat'),
        ]
    elif dataset == 'Cdataset':
        dataset_dir = os.path.join(data_root, 'Cdataset')
        candidates = [
            os.path.join(dataset_dir, 'Cdataset.mat'),
        ]
    else:
        dataset_dir = os.path.join(data_root, 'lrssl')
        candidates = [
            os.path.join(dataset_dir, 'lrssl.mat'),
        ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        'Could not find dataset MAT file. Tried:\n  ' + '\n  '.join(candidates) +
        '\nUse --mat_path to specify a file explicitly.'
    )


def load_mat(filepath):
    """Load Gdataset/Cdataset/lrssl MAT with optional drug_embedx/disease_embedx."""
    mat = scio.loadmat(filepath)
    required_keys = {'drug', 'disease', 'Wrname', 'Wdname', 'didr'}
    missing = required_keys - set(mat.keys())
    if missing:
        raise KeyError('{} missing required MAT keys: {}. Existing keys: {}'.format(filepath, missing, list(mat.keys())))

    drug_sim = np.asarray(mat['drug'], dtype=np.float32)
    disease_sim = np.asarray(mat['disease'], dtype=np.float32)
    if drug_sim.ndim != 2 or drug_sim.shape[0] != drug_sim.shape[1]:
        raise ValueError('drug similarity matrix must be square, got {}.'.format(drug_sim.shape))
    if disease_sim.ndim != 2 or disease_sim.shape[0] != disease_sim.shape[1]:
        raise ValueError('disease similarity matrix must be square, got {}.'.format(disease_sim.shape))

    drug_num = drug_sim.shape[0]
    disease_num = disease_sim.shape[0]
    drug_name = _extract_entity_names(mat['Wrname'], drug_num)
    disease_name = _extract_entity_names(mat['Wdname'], disease_num)

    interactions = np.asarray(mat['didr'], dtype=np.float32)
    expected_shape = (disease_num, drug_num)
    if interactions.shape != expected_shape:
        if interactions.T.shape == expected_shape:
            print('Warning: didr shape {} is drug x disease; transposing to disease x drug.'.format(interactions.shape))
            interactions = interactions.T
        else:
            raise ValueError('didr shape mismatch: got {}, expected disease x drug {}.'.format(interactions.shape, expected_shape))

    drug_embedx = _read_embedding_field(mat, 'drug_embedx', 'drug_embed', drug_num, 'drug_embedx')
    disease_embedx = _read_embedding_field(mat, 'disease_embedx', 'disease_embed', disease_num, 'disease_embedx')

    if drug_embedx is None and disease_embedx is None:
        pretrain_embeddings = None
    elif drug_embedx is None or disease_embedx is None:
        raise KeyError('Only one pretraining embedding field was found. Please include both drug_embedx and disease_embedx.')
    else:
        pretrain_embeddings = {
            'drug': drug_embedx,
            'disease': disease_embedx,
            'drug_dim': int(drug_embedx.shape[1]),
            'disease_dim': int(disease_embedx.shape[1]),
            'mat_path': filepath,
        }

    print('Loaded MAT dataset: {}'.format(filepath))
    print('  interactions/didr : {} = disease x drug'.format(interactions.shape))
    print('  drug similarity   : {}'.format(drug_sim.shape))
    print('  disease similarity: {}'.format(disease_sim.shape))
    if pretrain_embeddings is not None:
        print('  drug_embedx       : {}'.format(drug_embedx.shape))
        print('  disease_embedx    : {}'.format(disease_embedx.shape))

    return drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions, pretrain_embeddings


def get_train_adj(adj, train_mask, truth_label):
    for i in range(train_mask.shape[0]):
        for j in range(train_mask.shape[1]):
            if train_mask[i][j]:
                adj[i][j] = truth_label[i][j]
    return adj


def data_preparation(args):
    dataset = canonical_dataset_name(args.dataset)
    use_pretrain = bool(getattr(args, 'use_pretrain_embed', 1))
    data_root = getattr(args, 'data_root', './drug_data')

    mat_path = resolve_mat_path(dataset, getattr(args, 'mat_path', None), data_root=data_root)
    drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions, pretrain_embeddings = load_mat(mat_path)

    if use_pretrain and pretrain_embeddings is None:
        raise KeyError(
            'use_pretrain_embed=1, but no drug_embedx/disease_embedx were found in the loaded MAT file. '
            'Please rebuild/replace the MAT file with these fields, or run with --use_pretrain_embed 0.'
        )

    interactions = interactions.astype(np.float32)
    if not np.array_equal(interactions, interactions.astype(bool)):
        print('Warning: didr contains values other than 0/1; metrics and BCE expect binary labels.')

    seed = getattr(args, 'current_seed', None)
    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=seed)

    pos_row, pos_col = np.nonzero(interactions)
    neg_row, neg_col = np.nonzero(1 - interactions)
    assert len(pos_row) + len(neg_row) == np.prod(interactions.shape)

    train_data, test_data = [], []
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(
            kfold.split(pos_row), kfold.split(neg_row)):
        train_mask = np.zeros_like(interactions, dtype='bool')
        test_mask = np.zeros_like(interactions, dtype='bool')

        train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
        train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
        test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
        test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])

        train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
        test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
        train_mask[train_edge[0], train_edge[1]] = True
        test_mask[test_edge[0], test_edge[1]] = True
        train_data.append(train_mask)
        test_data.append(test_mask)

    disease_disease_sim_Matrix = get_disease_sim_Matrix(disease_sim, args.disease_TopK)
    drug_drug_sim_Matrix = get_drug_sim_Matrix(drug_sim, args.drug_TopK)
    truth_label = interactions
    pos_num = interactions.sum()
    neg_num = np.prod(interactions.shape) - pos_num
    if pos_num <= 0:
        raise ValueError('No positive disease-drug associations found in didr.')
    pos_weight = float(neg_num / pos_num)

    return (
        disease_disease_sim_Matrix,
        drug_drug_sim_Matrix,
        truth_label,
        train_data,
        test_data,
        pos_weight,
        pretrain_embeddings,
    )


class BatchManager(object):

    def __init__(self, data, batch_size, type):
        disease_input, drug_input, labels = [], [], []
        mask, label = data
        disease_drug_Adj = np.zeros((label.shape[0], label.shape[1]), dtype=np.float32)
        disease_drug_Adj = get_train_adj(disease_drug_Adj, mask, label)

        self.train_adj = disease_drug_Adj

        if type == 'train':
            train_mask, truth_label = data
            # Only use samples explicitly selected by train_mask for training.
            # Entries outside train_mask belong to validation/test folds and must not be
            # added as negative training samples.
            for i in range(len(train_mask)):
                for j in range(len(train_mask[0])):
                    if train_mask[i][j]:
                        disease_input.append(i)
                        drug_input.append(j)
                        labels.append(truth_label[i][j])
            num_batch = int(math.ceil(len(disease_input) / batch_size))
            self.batch_data = list()

            for i in range(num_batch):
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                label_batch = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append([input_disease, input_drug, label_batch])

        elif type == 'test':
            test_mask, truth_label = data
            for i in range(len(test_mask)):
                for j in range(len(test_mask[0])):
                    if test_mask[i][j]:
                        disease_input.append(i)
                        drug_input.append(j)
                        labels.append(truth_label[i][j])
            num_batch = int(math.ceil(len(disease_input) / batch_size))
            self.batch_data = list()
            for i in range(num_batch):
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                label_batch = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append([input_disease, input_drug, label_batch])
        else:
            raise ValueError('Unknown BatchManager type: {}'.format(type))

        self.len_data = len(self.batch_data)

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def data_split(config, train_mask, test_mask, original_interactions):
    train_data = (train_mask, original_interactions)
    test_data = (test_mask, original_interactions)
    train_manager = BatchManager(train_data, config['batch_size'], 'train')
    test_manager = BatchManager(test_data, config['batch_size'], 'test')

    return train_manager, test_manager
