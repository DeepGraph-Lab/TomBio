import argparse
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from loader import *
from model import DRDM
from sklearn import metrics
from torch.optim.lr_scheduler import CyclicLR

sys.path.append('.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Run DRDM ablation with Explicit Topology Decoder / TTM-D.')
    parser.add_argument('--dataset', nargs='?', default='Gdataset', help='Choose a dataset. [Gdataset/Fdataset/Cdataset/lrssl]')
    parser.add_argument('--data_root', type=str, default='./drug_data', help='Root directory containing Gdataset, Cdataset and lrssl folders.')
    parser.add_argument('--mat_path', type=str, default=None, help='Optional explicit path to dataset MAT file.')
    parser.add_argument('--epochs', type=int, default=130, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024 * 5, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.005, help='init Learning rate.')
    parser.add_argument('--embedding_size', type=int, default=768)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--disease_TopK', type=int, default=4)
    parser.add_argument('--drug_TopK', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0, help='Starting seed. Original effective trial seeds are 0,1,2,...')
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--exp_coff', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.6, help='the coefficient of feature fusion')
    parser.add_argument('--wr', type=float, default=0.6, help='the coefficient of feature fusion')

    # Pretrained node embedding options.
    parser.add_argument('--use_pretrain_embed', type=int, default=1,#使用预训练
                        help='1: initialize nodes from drug_embedx/disease_embedx in MAT; 0: original random init.')
    parser.add_argument('--pretrain_freeze', type=int, default=0,
                        help='1: freeze pretrained node embeddings; 0: fine-tune them with DRDM.')
    parser.add_argument('--pretrain_init_scale', type=float, default=0.5,
                        help='L2-normalized pretrained init scale after dimension adaptation.')
    parser.add_argument('--embedding_lr_mult', type=float, default=0.2,
                        help='Learning-rate multiplier for pretrained disease/drug embedding parameters.')

    # Neighbor self-gating switch.
    # Contrastive learning, embedding noise, topology-prior residual,
    # non-legacy propagation modes, and learnable decoder-scale gates are removed.
    parser.add_argument('--ablate_neighbor_self_gating', type=int, default=1,#关掉门控
                        help='1: make neighbor graph branch use the same initial embeddings as association branch; 0: use self-gated neighbor embeddings.')

    # Explicit Topology Decoder / TTM-D options copied from drdm57x.
    parser.add_argument('--use_topology_decoder', type=int, default=1,#使用拓扑解码器
                        help='1: enable explicit topology decoder / TTM-D; 0: use original dot-product decoder.')
    parser.add_argument('--use_ttm', type=int, default=None,
                        help='Alias of --use_topology_decoder for compatibility with drdm57x. If set, overrides --use_topology_decoder.')
    parser.add_argument('--topo_hidden', type=int, default=256,
                        help='Hidden dimension of topology encoders and relation decoder.')
    parser.add_argument('--topo_alpha', type=float, default=0.1,#0.1
                        help='Strength of topology embedding injection.')
    parser.add_argument('--rel_alpha', type=float, default=0.15,#0.15
                        help='Strength of relation MLP decoder logit.')
    parser.add_argument('--topo_loss_weight', type=float, default=0.5,#0.5
                        help='Weight of auxiliary topology compression loss.')
    parser.add_argument('--degree_tau', type=float, default=5.0,
                        help='Temperature for degree-aware topology injection gate.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout used in propagation and topology decoder MLPs.')
    # Result logging.
    parser.add_argument('--results_dir', type=str, default='./results18', help='Directory for metric CSV/TXT logs.')
    parser.add_argument('--results_prefix', type=str, default=None, help='Optional prefix for result files.')
    return parser.parse_args()



def config_model():
    config = OrderedDict()
    config['dataset'] = args.dataset
    config['data_root'] = args.data_root
    config['mat_path'] = args.mat_path
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['embedding_size'] = args.embedding_size
    config['n_layers'] = args.n_layers
    config['weight_decay'] = args.weight_decay
    config['disease_TopK'] = args.disease_TopK
    config['drug_TopK'] = args.drug_TopK
    config['seed'] = args.seed
    config['n_splits'] = args.n_splits
    config['device'] = device
    config['exp_coff'] = args.exp_coff
    config['wd'] = args.wd
    config['wr'] = args.wr
    config['use_pretrain_embed'] = bool(args.use_pretrain_embed)
    config['pretrain_freeze'] = bool(args.pretrain_freeze)
    config['pretrain_init_scale'] = args.pretrain_init_scale
    config['embedding_lr_mult'] = args.embedding_lr_mult
    config['ablate_neighbor_self_gating'] = bool(args.ablate_neighbor_self_gating)
    use_topology_decoder = args.use_topology_decoder if args.use_ttm is None else args.use_ttm
    config['use_topology_decoder'] = bool(use_topology_decoder)
    config['use_ttm'] = bool(use_topology_decoder)  # compatibility name used by drdm57x
    config['topo_hidden'] = args.topo_hidden
    config['topo_alpha'] = args.topo_alpha
    config['rel_alpha'] = args.rel_alpha
    config['topo_loss_weight'] = args.topo_loss_weight
    config['degree_tau'] = args.degree_tau
    config['dropout'] = args.dropout
    config['prior_mode'] = 'none'
    config['prop_weight_mode'] = 'legacy'
    return config


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def safe_binary_metrics(labels, scores):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(np.unique(labels)) < 2:
        return float('nan'), float('nan')
    aupr = metrics.average_precision_score(y_true=labels, y_score=scores)
    auroc = metrics.roc_auc_score(y_true=labels, y_score=scores)
    return float(auroc), float(aupr)


def format_seconds(seconds):
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return '{}h {}m {:.2f}s'.format(h, m, s)


def evaluate(model, test_manager):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in test_manager.iter_batch():
            score, label = model.predict(batch)
            scores.append(score.cpu().detach().numpy())
            labels.append(np.asarray(label))
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    auroc, aupr = safe_binary_metrics(labels, scores)
    return scores, labels, auroc, aupr


def build_optimizer_and_lr_bounds(model, args):
    """Use a smaller LR for pretrained node embeddings.

    This protects the pretrained semantic space from being overwritten too
    aggressively by topology-decoder losses, which helps preserve AUROC.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError('No trainable model parameters found.')

    use_split_lr = bool(args.use_pretrain_embed) and (not bool(args.pretrain_freeze)) and args.embedding_lr_mult != 1.0
    if not use_split_lr:
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        return optimizer, 0.1 * args.lr, args.lr

    embedding_params = []
    embedding_params.extend([p for p in model.disease_embedding.parameters() if p.requires_grad])
    embedding_params.extend([p for p in model.drug_embedding.parameters() if p.requires_grad])
    embedding_ids = set(id(p) for p in embedding_params)
    other_params = [p for p in trainable_params if id(p) not in embedding_ids]

    param_groups = []
    base_lrs = []
    max_lrs = []
    if embedding_params:
        embed_lr = args.lr * args.embedding_lr_mult
        param_groups.append({'params': embedding_params, 'lr': embed_lr})
        base_lrs.append(0.1 * embed_lr)
        max_lrs.append(embed_lr)
    if other_params:
        param_groups.append({'params': other_params, 'lr': args.lr})
        base_lrs.append(0.1 * args.lr)
        max_lrs.append(args.lr)

    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    return optimizer, base_lrs, max_lrs


def make_result_paths(args):
    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    use_topology_decoder = args.use_topology_decoder if args.use_ttm is None else args.use_ttm
    ablation_tag = 'gate{}_topo{}'.format(
        0 if bool(args.ablate_neighbor_self_gating) else 1,
        1 if bool(use_topology_decoder) else 0,
    )
    prefix = args.results_prefix or '{}_drdm_{}_{}_{}'.format(
        args.dataset,
        'pretrain' if bool(args.use_pretrain_embed) else 'random',
        ablation_tag,
        stamp,
    )
    return {
        'epoch': os.path.join(args.results_dir, '{}_epoch_metrics.csv'.format(prefix)),
        'fold': os.path.join(args.results_dir, '{}_fold_metrics.csv'.format(prefix)),
        'trial': os.path.join(args.results_dir, '{}_trial_summary.csv'.format(prefix)),
        'txt': os.path.join(args.results_dir, '{}_summary.txt'.format(prefix)),
    }


def save_results(result_paths, epoch_records, fold_records, trial_records, final_report_lines):
    pd.DataFrame(epoch_records).to_csv(result_paths['epoch'], index=False, encoding='utf-8-sig')
    pd.DataFrame(fold_records).to_csv(result_paths['fold'], index=False, encoding='utf-8-sig')
    pd.DataFrame(trial_records).to_csv(result_paths['trial'], index=False, encoding='utf-8-sig')
    with open(result_paths['txt'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_report_lines) + '\n')


args = parse_args()
config = config_model()


if __name__ == '__main__':
    result_paths = make_result_paths(args)
    total_start_time = time.perf_counter()
    epoch_records, fold_records, trial_records = [], [], []
    avg_final_auroc, avg_final_aupr = [], []
    avg_trial_time_sec = []

    print('Device:', device)
    print('Dataset:', args.dataset)
    print('Data root:', args.data_root)
    print('Use pretrained node embeddings:', bool(args.use_pretrain_embed))
    print('Embedding LR multiplier:', args.embedding_lr_mult)
    print('Removed modules: contrastive_learning, embedding_noise, topology_prior_residual, '
          'non_legacy_prop_weight_modes, learnable_decoder_scale')
    print('Neighbor self-gating:')
    print('  ablate_neighbor_self_gating:', bool(args.ablate_neighbor_self_gating))
    use_topology_decoder = args.use_topology_decoder if args.use_ttm is None else args.use_ttm
    print('Explicit Topology Decoder / TTM-D:')
    print('  use_topology_decoder:', bool(use_topology_decoder))
    print('  topo_hidden:', args.topo_hidden)
    print('  topo_alpha:', args.topo_alpha)
    print('  rel_alpha:', args.rel_alpha)
    print('  prior_mode: none (fixed; prior residual removed)')
    print('  topo_loss_weight:', args.topo_loss_weight)
    print('  degree_tau:', args.degree_tau)
    print('  prop_weight_mode: legacy (fixed)')
    print('Result files will be saved to:')
    for key, value in result_paths.items():
        print('  {}: {}'.format(key, value))

    for i in range(args.num_trials):
        trial_start_time = time.perf_counter()
        trial_seed = args.seed + i
        args.current_seed = trial_seed
        setup_seed(trial_seed)

        (
            disease_adj,
            drug_adj,
            original_interactions,
            all_train_mask,
            all_test_mask,
            pos_weight,
            pretrain_embeddings,
        ) = data_preparation(args)

        final_scores_all, final_labels_all = [], []
        print('+++++++++++++++ This is {}-th {}-fold validation. seed={} +++++++++++++++'.format(
            i + 1, args.n_splits, trial_seed
        ))

        for fold_num in range(len(all_train_mask)):
            fold_start_time = time.perf_counter()
            print('--------------- This is fold {}/{}. ---------------'.format(fold_num + 1, len(all_train_mask)))
            train_manager, test_manager = data_split(
                config,
                all_train_mask[fold_num],
                all_test_mask[fold_num],
                original_interactions,
            )
            train_adj = train_manager.train_adj

            model = DRDM(config, (train_manager, train_adj, disease_adj, drug_adj, pos_weight, pretrain_embeddings)).to(device)
            optimizer, base_lrs, max_lrs = build_optimizer_and_lr_bounds(model, args)
            lr_scheduler = CyclicLR(
                optimizer,
                base_lr=base_lrs,
                max_lr=max_lrs,
                step_size_up=2000,
                mode='exp_range',
                gamma=0.995,
                cycle_momentum=False,
            )

            final_fold = None

            for epoch in range(args.epochs):
                epoch_start_time = time.perf_counter()
                model.train()
                loss_list = []
                for batch in train_manager.iter_batch(shuffle=True):
                    loss, _ = model.forward(batch, True)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    loss_list.append(float(loss.item()))

                scores, labels, auroc, aupr = evaluate(model, test_manager)
                loss_mean = float(np.mean(loss_list)) if loss_list else float('nan')
                epoch_time_sec = time.perf_counter() - epoch_start_time
                print('Epoch: {}, loss: {:.6f}, auroc: {:.6f}, aupr: {:.6f}, time: {}'.format(
                    epoch + 1, loss_mean, auroc, aupr, format_seconds(epoch_time_sec)
                ))

                epoch_records.append({
                    'trial': i + 1,
                    'seed': trial_seed,
                    'fold': fold_num + 1,
                    'epoch': epoch + 1,
                    'loss': loss_mean,
                    'auroc': auroc,
                    'aupr': aupr,
                    'ablate_neighbor_self_gating': bool(args.ablate_neighbor_self_gating),
                    'use_topology_decoder': bool(use_topology_decoder),
                    'topo_alpha': args.topo_alpha,
                    'rel_alpha': args.rel_alpha,
                    'topo_loss_weight': args.topo_loss_weight,
                    'degree_tau': args.degree_tau,
                    'embedding_lr_mult': args.embedding_lr_mult,
                    'epoch_time_sec': epoch_time_sec,
                })

                if (epoch + 1) == args.epochs:
                    final_fold = {
                        'epoch': epoch + 1,
                        'auroc': auroc,
                        'aupr': aupr,
                        'scores': scores.copy(),
                        'labels': labels.copy(),
                    }

            fold_time_sec = time.perf_counter() - fold_start_time
            final_scores_all.append(final_fold['scores'])
            final_labels_all.append(final_fold['labels'])
            fold_records.append({
                'trial': i + 1,
                'seed': trial_seed,
                'fold': fold_num + 1,
                'final_epoch': final_fold['epoch'],
                'final_auroc': final_fold['auroc'],
                'final_aupr': final_fold['aupr'],
                'ablate_neighbor_self_gating': bool(args.ablate_neighbor_self_gating),
                'use_topology_decoder': bool(use_topology_decoder),
                'topo_alpha': args.topo_alpha,
                'rel_alpha': args.rel_alpha,
                'topo_loss_weight': args.topo_loss_weight,
                'degree_tau': args.degree_tau,
                'fold_time_sec': fold_time_sec,
            })
            print(
                'Fold {} final_auroc={:.5f}, final_aupr={:.5f}; time={}'.format(
                    fold_num + 1,
                    final_fold['auroc'], final_fold['aupr'],
                    format_seconds(fold_time_sec)
                )
            )

        final_scores_all = np.concatenate(final_scores_all)
        final_labels_all = np.concatenate(final_labels_all)
        final_auroc, final_aupr = safe_binary_metrics(final_labels_all, final_scores_all)

        trial_time_sec = time.perf_counter() - trial_start_time

        avg_final_auroc.append(final_auroc)
        avg_final_aupr.append(final_aupr)
        avg_trial_time_sec.append(trial_time_sec)

        trial_records.append({
            'trial': i + 1,
            'seed': trial_seed,
            'final_auroc': final_auroc,
            'final_aupr': final_aupr,
            'ablate_neighbor_self_gating': bool(args.ablate_neighbor_self_gating),
            'use_topology_decoder': bool(use_topology_decoder),
            'topo_alpha': args.topo_alpha,
            'rel_alpha': args.rel_alpha,
            'topo_loss_weight': args.topo_loss_weight,
            'degree_tau': args.degree_tau,
            'trial_time_sec': trial_time_sec,
            'trial_time_hms': format_seconds(trial_time_sec),
        })

        print('------------------------------------------------------------------------')
        print('{}-th {}-cv final_auroc: {:.5f}'.format(i + 1, args.n_splits, final_auroc))
        print('{}-th {}-cv final_aupr : {:.5f}'.format(i + 1, args.n_splits, final_aupr))
        print('{}-th {}-cv trial_time: {}'.format(i + 1, args.n_splits, format_seconds(trial_time_sec)))

        # Save after every trial to avoid losing long-run results.
        save_results(result_paths, epoch_records, fold_records, trial_records, ['Intermediate results saved.'])

    total_time_sec = time.perf_counter() - total_start_time
    final_report_lines = [
        '------------------------------------------------------------------------',
        'final_auroc mean: {:.5f}, std: {:.5f}'.format(np.nanmean(avg_final_auroc), np.nanstd(avg_final_auroc)),
        'final_aupr  mean: {:.5f}, std: {:.5f}'.format(np.nanmean(avg_final_aupr), np.nanstd(avg_final_aupr)),
        'trial_time mean: {}, std_sec: {:.2f}'.format(
            format_seconds(np.nanmean(avg_trial_time_sec)), np.nanstd(avg_trial_time_sec)
        ),
        'total_time: {}, total_time_sec: {:.2f}'.format(format_seconds(total_time_sec), total_time_sec),
        '',
        'epoch_metrics_csv: {}'.format(result_paths['epoch']),
        'fold_metrics_csv : {}'.format(result_paths['fold']),
        'trial_summary_csv: {}'.format(result_paths['trial']),
    ]
    for line in final_report_lines:
        print(line)

    save_results(result_paths, epoch_records, fold_records, trial_records, final_report_lines)
