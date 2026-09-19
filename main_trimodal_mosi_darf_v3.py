"""
三模态训练脚本 - 用于 MOSEI/MOSI 数据集
实现三模态 beta 计算逻辑:
- 最强模态 beta: gap = (score_最强 - score_次强 + score_最强 - score_最弱) / 2
- 次强模态 beta: gap = (score_次强 - score_最弱)
- 最弱模态 beta = 0

模型架构:
- Text: Transformer 编码器 (来自 mosei 项目)
- Audio: MLP 编码器
- Visual: MLP 编码器

训练设置 (参考 mosei 项目):
- batch_size: 32
- lr: 0.001
- n_epochs: 30
- optimizer: SGD (momentum=0.9, weight_decay=1e-4)
- scheduler: cosine_schedule_with_warmup (warmup=2.5 epochs)
"""

import argparse
import os
import sys
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
for ROOT_DIR in [
    os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')),
    os.path.abspath(os.path.join(SCRIPT_DIR, '..')),
]:
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
import copy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
import pickle
import csv

# Cosine scheduler with warmup (from transformers library)
try:
    from transformers import get_cosine_schedule_with_warmup
except ImportError:
    # 如果没有安装 transformers，使用简单的 cosine scheduler
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
        from torch.optim.lr_scheduler import LambdaLR
        import math
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return LambdaLR(optimizer, lr_lambda, last_epoch)

# 支持两种模型: 简单的 TriModalClassifier 或 Transformer-based 的 TriModalTransformer
from models.basic_model import TriModalClassifier
from models.transformers import TriModalTransformer
from models.contrastive_loss import CrossModalContrastiveLoss
from models.focal_loss import FocalLoss
from utils.utils import setup_seed, weight_init
from dataset.MOSEIDataset import MOSEI_dataset,get_mosei_dataloaders

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MOSI', type=str,
                        choices=['MOSEI', 'MOSI'])
    parser.add_argument('--modulation', default='Ours', type=str,
                        choices=['Normal', 'OGM', 'Ours'])
    parser.add_argument('--data_path', default='./data/CMU-MOSI/Processed', type=str)
    parser.add_argument('--dataset_name', default='unaligned_50', type=str)
    parser.add_argument('--trimodal_loss_protocol', default='paper', type=str,
                        choices=['paper', 'legacy'],
                        help='paper: 仅用融合 CE 作为基础训练目标；legacy: 融合+单模态 CE')
    parser.add_argument('--trimodal_label_protocol', default='paper_threshold', type=str,
                        choices=['paper_threshold', 'classification'],
                        help='paper_threshold: 用连续情感值按0.5阈值转三分类；classification: 直接使用现成三分类标签')
    parser.add_argument('--trimodal_eval_protocol', default='paper_table6', type=str,
                        choices=['paper_table6', 'overall_acc'],
                        help='paper_table6: 用 balanced accuracy 作为论文表格中的 Accuracy；overall_acc: 用普通准确率')

    # ========== 训练超参数 (参考 mosei 项目) ==========
    parser.add_argument('--batch_size', default=32, type=int)  # mosei: 32
    parser.add_argument('--epochs', default=30, type=int)       # mosei: 30
    parser.add_argument('--n_classes', default=3, type=int)
    parser.add_argument('--data_workers', default=8, type=int)

    # 模态特征维度（当前成功复现 MOSI 论文结果所使用的 unaligned_50 处理包）
    parser.add_argument('--text_dim', default=768, type=int)
    parser.add_argument('--audio_dim', default=5, type=int)
    parser.add_argument('--visual_dim', default=20, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int, help='MLP hidden dim')
    parser.add_argument('--embed_dim', default=120, type=int, help='embedding dimension (mosei: 120)')

    # Transformer 配置 (参考 mosei: depth=4, num_heads=10, embed_dim=120)
    parser.add_argument('--transformer_depth', default=4, type=int, help='Transformer layers')
    parser.add_argument('--transformer_heads', default=10, type=int, help='attention heads (mosei: 10)')
    parser.add_argument('--mlp_ratio', default=4.0, type=float, help='MLP ratio in Transformer')
    parser.add_argument('--drop_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--fusion_method', default='concat', type=str, choices=['concat', 'gated'])

    # 模型选择
    parser.add_argument('--model_type', default='transformer', type=str,
                        choices=['simple', 'transformer'],
                        help='simple: TriModalClassifier, transformer: TriModalTransformer')

    # ========== 优化器设置 (参考 mosei: SGD, lr=0.001, momentum=0.9, weight_decay=1e-4) ==========
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])  # mosei: SGD
    parser.add_argument('--learning_rate', default=0.001, type=float)  # mosei: 0.001
    parser.add_argument('--momentum', default=0.9, type=float)         # mosei: 0.9
    parser.add_argument('--weight_decay', default=1e-4, type=float)    # mosei: 1e-4
    parser.add_argument('--warmup_epochs', default=2.5, type=float, help='warmup epochs for cosine scheduler')

    parser.add_argument('--modulation_starts', default=0, type=int)
    parser.add_argument('--modulation_ends', default=30, type=int)  # 与 epochs 对齐
    parser.add_argument('--alpha', default=0.8, type=float, help='proximal 正则化系数')
    parser.add_argument('--beta_scale', default=0.8, type=float, help='beta 缩放系数')
    parser.add_argument('--k_threshold', default=0.1, type=float, help='prime learning window 的 k 值阈值')

    # ========== 三个创新点 ==========
    parser.add_argument('--use_cmob', action='store_true', help='使用三模态 CMoB 风格因果模态估值')
    parser.add_argument('--cmob_weight', default=1.0, type=float, help='CMoB 风格单模态重加权强度')
    parser.add_argument('--use_contrastive', action='store_true', help='使用三模态对比学习')
    parser.add_argument('--con_weight', default=0.1, type=float, help='对比学习损失权重')
    parser.add_argument('--con_start_epoch', default=0, type=int, help='对比学习启用起始 epoch')
    parser.add_argument('--use_sample_weighting', action='store_true', help='使用样本级加权 Focal Loss')
    parser.add_argument('--sample_weighting_scope', default='all', type=str, choices=['all', 'fusion_only', 'unimodal_only'],
                        help='Focal Loss 作用范围: all=融合与单模态都用, fusion_only=只作用于融合, unimodal_only=只作用于单模态')
    parser.add_argument('--focal_gamma_start', default=1.0, type=float, help='Focal Loss 起始 gamma')
    parser.add_argument('--focal_gamma_end', default=2.0, type=float, help='Focal Loss 结束 gamma')
    parser.add_argument('--mosi_adapt_cmob_cap', default=0.45, type=float, help='MOSI ????? CMoB ????')
    parser.add_argument('--mosi_adapt_focal_start_cap', default=0.0, type=float, help='MOSI ????? focal gamma ????')
    parser.add_argument('--mosi_adapt_focal_end_cap', default=0.55, type=float, help='MOSI ????? focal gamma ????')
    parser.add_argument('--phsf_aux_weight', default=0.0, type=float,
                        help='Extra PHSF auxiliary unimodal loss weight when CMDR is disabled.')
    parser.add_argument('--phsf_aux_start_epoch', default=0, type=int,
                        help='Start epoch of the extra PHSF auxiliary loss.')
    parser.add_argument('--phsf_aux_targets', default='non_text', type=str,
                        choices=['all', 'non_text'],
                        help='Modal branches used by the extra PHSF auxiliary loss.')
    parser.add_argument('--phsf_aux_include_cmdr', action='store_true',
                        help='Also apply the extra PHSF auxiliary loss when CMDR/use_cmob is enabled.')

    # ========== 弱模态平衡框架 ==========
    parser.add_argument('--use_branch_specialist', action='store_true', help='启用 tri-modal branch specialist')
    parser.add_argument('--bs_start_epoch', default=0, type=int, help='branch specialist 启用 epoch')
    parser.add_argument('--bs_target', default='weakest', type=str,
                        help='branch specialist 作用模态: weakest/all/non_text/text/audio/visual/audio,visual')
    parser.add_argument('--bs_temperature', default=2.0, type=float, help='branch specialist KD 温度')
    parser.add_argument('--bs_shared_ce_weight', default=0.2, type=float, help='shared CE 权重')
    parser.add_argument('--bs_specialist_ce_weight', default=0.4, type=float, help='specialist CE 权重')
    parser.add_argument('--bs_kd_weight', default=0.2, type=float, help='specialist KD 权重')
    parser.add_argument('--bs_boost_only_ce', action='store_true', help='specialist CE only supervises boost head')
    parser.add_argument('--bs_detach_base_for_kd', action='store_true', help='detach base logits when computing specialist KD')
    parser.add_argument('--bs_min_teacher_conf', default=0.0, type=float, help='minimum teacher confidence to enable specialist loss')
    parser.add_argument('--bs_eval_target', default='', type=str,
                        help='??? specialist ??????; ????? bs_target')
    parser.add_argument('--bs_eval_text_blend', default=0.0, type=float, help='??? text specialist ????')
    parser.add_argument('--bs_eval_audio_blend', default=0.0, type=float, help='??? audio specialist ????')
    parser.add_argument('--bs_eval_visual_blend', default=0.0, type=float, help='??? visual specialist ????')
    parser.add_argument('--bs_eval_fusion_weight', default=0.0, type=float, help='???? specialist ????? fusion ???')
    parser.add_argument('--bs_fusion_aux_weight', default=0.0, type=float, help='??? specialist ???? fusion ??? CE ??')
    parser.add_argument('--bs_fusion_residual_scale', default=0.0, type=float, help='??? specialist ???? fusion logits ?????')
    parser.add_argument('--bs_use_feature_repair', action='store_true', help='?? feature-level specialist repair')
    parser.add_argument('--bs_feature_residual_scale', default=0.0, type=float, help='feature repair residual ????')
    parser.add_argument('--bs_feature_modal_ce_weight', default=0.0, type=float, help='feature repair ??? CE ??')
    parser.add_argument('--bs_feature_fusion_ce_weight', default=0.0, type=float, help='feature repair ?? CE ??')
    parser.add_argument('--use_single_run_repair', action='store_true', help='启用 tri-modal single-run repair')
    parser.add_argument('--sr_start_epoch', default=0, type=int, help='single-run repair 启用起始 epoch')
    parser.add_argument('--sr_end_epoch', default=29, type=int, help='single-run repair 结束 epoch')
    parser.add_argument('--sr_target_pool', default='all', type=str, choices=['all', 'non_text'],
                        help='single-run repair 选择弱模态的候选池: all=文本/音频/视觉都参与, non_text=只在音频/视觉中选择')
    parser.add_argument('--sr_temperature', default=2.0, type=float, help='single-run repair KD 温度')
    parser.add_argument('--sr_ce_weight', default=0.2, type=float, help='repair 共享头 CE 权重')
    parser.add_argument('--sr_specialist_ce_weight', default=0.3, type=float, help='repair specialist CE 权重')
    parser.add_argument('--sr_fused_ce_weight', default=0.3, type=float, help='repair fused CE 权重')
    parser.add_argument('--sr_kd_weight', default=0.2, type=float, help='repair KD 权重')
    parser.add_argument('--sr_gap_scale', default=4.0, type=float, help='repair 权重的 gap 放大系数')
    parser.add_argument('--sr_use_fusion_residual', action='store_true', help='???? specialist ????? repair fusion ??')
    parser.add_argument('--sr_fusion_residual_scale', default=0.0, type=float, help='single-run repair ? specialist ????? fusion logits ?????')

    parser.add_argument('--ckpt_path', default='./ckpt', type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--use_tensorboard', default=True, type=bool)
    parser.add_argument('--tensorboard_path', default='./logs', type=str)
    parser.add_argument('--random_seed', default=2024, type=int)  # mosei: 2024
    parser.add_argument('--gpu_ids', default='0', type=str)

    return parser.parse_args()


def resolve_paper_like_trimodal_chain(args):
    """
    将 MOSI 的默认链路切到当前最接近论文 Table 6 的处理包。

    说明：
    - 当前仓库历史上混入过多份 MOSI 处理结果；
    - 推荐使用 data/CMU-MOSI/Processed 下的 unaligned_50.pkl；
    - 特征维度为 768/5/20，并包含 regression_labels/classification_labels。
    """
    if (
        args.dataset == 'MOSI'
        and os.path.basename(os.path.normpath(args.data_path)) == 'Processed'
        and args.dataset_name == 'unaligned_50'
    ):
        args.dataset_name = 'unaligned_50'
        args.text_dim = 768
        args.audio_dim = 5
        args.visual_dim = 20
        args.trimodal_loss_protocol = 'paper'
        args.trimodal_label_protocol = 'paper_threshold'
        args.trimodal_eval_protocol = 'paper_table6'
        args.beta_scale = 0.9
        args.k_threshold = 0.04
        print('[PaperChain] 已自动切换到当前最接近论文 Table 6 的 MOSI 链路：')
        print(f'  data_path={args.data_path}')
        print(f'  dataset_name={args.dataset_name}')
        print(f'  text_dim={args.text_dim}')
        print(f'  audio_dim={args.audio_dim}')
        print(f'  visual_dim={args.visual_dim}')
        print(f'  trimodal_loss_protocol={args.trimodal_loss_protocol}')
        print(f'  trimodal_label_protocol={args.trimodal_label_protocol}')
        print(f'  trimodal_eval_protocol={args.trimodal_eval_protocol}')
        print(f'  beta_scale={args.beta_scale}')
        print(f'  k_threshold={args.k_threshold}')
    return args


def compute_trimodal_beta(score_t, score_a, score_v, alpha=0.95):
    """
    计算三模态的 beta 值

    逻辑:
    - 先判断出最强、次强和最弱的模态
    - 最强模态: gap = (score_最强 - score_次强 + score_最强 - score_最弱) / 2
    - 次强模态: gap = (score_次强 - score_最弱)
    - 最弱模态: beta = 0

    Args:
        score_t: 文本模态的 score (sum of softmax probabilities at true labels)
        score_a: 音频模态的 score
        score_v: 视觉模态的 score
        alpha: 缩放系数

    Returns:
        beta_t, beta_a, beta_v: 三个模态的 beta 值
    """
    tanh = torch.tanh

    # 将 scores 放入列表进行排序
    scores = [
        ('text', score_t),
        ('audio', score_a),
        ('visual', score_v)
    ]

    # 按 score 降序排序
    scores_sorted = sorted(scores, key=lambda x: x[1].item(), reverse=True)

    strongest_name, strongest_score = scores_sorted[0]
    middle_name, middle_score = scores_sorted[1]
    weakest_name, weakest_score = scores_sorted[2]

    # 计算各模态的 gap 和 beta
    # 最强模态: gap = (score_最强 - score_次强 + score_最强 - score_最弱) / 2
    gap_strongest = (strongest_score - middle_score + strongest_score - weakest_score) / 2
    beta_strongest = alpha * torch.exp(tanh(gap_strongest))

    # 次强模态: gap = (score_次强 - score_最弱)
    gap_middle = middle_score - weakest_score
    beta_middle = alpha * torch.exp(tanh(gap_middle))

    # 最弱模态: beta = 0
    beta_weakest = torch.tensor(0.0).to(score_t.device)

    # 按原始顺序返回 beta 值
    beta_dict = {
        strongest_name: beta_strongest,
        middle_name: beta_middle,
        weakest_name: beta_weakest
    }

    return beta_dict['text'], beta_dict['audio'], beta_dict['visual']


def get_model_module(model):
    return model.module if hasattr(model, 'module') else model


def infer_trimodal_feature_dims(train_loader, args):
    dataset = train_loader.dataset
    inferred_text_dim = int(dataset.text.shape[-1])
    inferred_audio_dim = int(dataset.audio.shape[-1])
    inferred_visual_dim = int(dataset.vision.shape[-1])

    if (args.text_dim, args.audio_dim, args.visual_dim) != (inferred_text_dim, inferred_audio_dim, inferred_visual_dim):
        print(
            f"[AutoInfer] 特征维度已自动校正: "
            f"text {args.text_dim}->{inferred_text_dim}, "
            f"audio {args.audio_dim}->{inferred_audio_dim}, "
            f"visual {args.visual_dim}->{inferred_visual_dim}"
        )
    args.text_dim = inferred_text_dim
    args.audio_dim = inferred_audio_dim
    args.visual_dim = inferred_visual_dim
    return args


def compute_macro_f1(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0

    f1_scores = []
    for class_idx in range(n_classes):
        tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
        fp = np.sum((y_true != class_idx) & (y_pred == class_idx))
        fn = np.sum((y_true == class_idx) & (y_pred != class_idx))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def compute_balanced_accuracy(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0

    recalls = []
    for class_idx in range(n_classes):
        class_mask = (y_true == class_idx)
        class_total = np.sum(class_mask)
        if class_total == 0:
            continue
        class_correct = np.sum(class_mask & (y_pred == class_idx))
        recalls.append(class_correct / class_total)

    return float(np.mean(recalls)) if len(recalls) > 0 else 0.0


def summarize_classification_metrics(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    overall_acc = float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0
    paper_acc = compute_balanced_accuracy(y_true, y_pred, n_classes)
    macro_f1 = compute_macro_f1(y_true, y_pred, n_classes)
    return {
        'overall_acc': overall_acc,
        'paper_acc': paper_acc,
        'macro_f1': macro_f1,
    }


def resolve_trimodal_primary_metric(args, metric_dict):
    if getattr(args, 'trimodal_eval_protocol', 'paper_table6') == 'paper_table6':
        return metric_dict['paper_acc']
    return metric_dict['overall_acc']


def resolve_eval_checkpoint_path(args):
    if os.path.isfile(args.ckpt_path):
        return args.ckpt_path
    if os.path.isdir(args.ckpt_path):
        candidates = [
            os.path.join(args.ckpt_path, name)
            for name in os.listdir(args.ckpt_path)
            if name.endswith('.pth') and f'best_model_{args.dataset}_' in name
        ]
        if not candidates:
            return None
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]
    return None


def compute_cls_loss(logits, label, ce_criterion, focal_criterion=None):
    if focal_criterion is not None:
        return focal_criterion(logits, label)
    return ce_criterion(logits, label)


def compute_label_confidence(logits, label):
    prob = F.softmax(logits.detach(), dim=1)
    conf_batch = prob.gather(1, label.view(-1, 1)).squeeze(1)
    return conf_batch.mean().item(), conf_batch


def compute_trimodal_causal_weights(model, t_feat, a_feat, v_feat, out_fusion, label, criterion):
    # Trimodal CMDR-style extension: criterion uses mean reduction, so TE values
    # are mini-batch scalar dominance/diagnostic signals, not per-sample CMoB
    # modality valuations. They are used for batch-level loss reweighting only.
    module = get_model_module(model)
    if not hasattr(module, 'get_fusion_output_from_features'):
        return {'text': 1.0, 'audio': 1.0, 'visual': 1.0}, None

    with torch.no_grad():
        zero_t = torch.zeros_like(t_feat)
        zero_a = torch.zeros_like(a_feat)
        zero_v = torch.zeros_like(v_feat)

        joint_loss = criterion(out_fusion.detach(), label)
        loss_wo_t = criterion(module.get_fusion_output_from_features(zero_t, a_feat.detach(), v_feat.detach()), label)
        loss_wo_a = criterion(module.get_fusion_output_from_features(t_feat.detach(), zero_a, v_feat.detach()), label)
        loss_wo_v = criterion(module.get_fusion_output_from_features(t_feat.detach(), a_feat.detach(), zero_v), label)

        te_text = (loss_wo_t - joint_loss).item()
        te_audio = (loss_wo_a - joint_loss).item()
        te_visual = (loss_wo_v - joint_loss).item()
        te_tensor = torch.tensor([te_text, te_audio, te_visual], device=out_fusion.device, dtype=out_fusion.dtype)
        balance_weights = torch.softmax(-te_tensor, dim=0) * 3.0

    return {
        'text': float(balance_weights[0].item()),
        'audio': float(balance_weights[1].item()),
        'visual': float(balance_weights[2].item()),
    }, {
        'te_text': te_text,
        'te_audio': te_audio,
        'te_visual': te_visual,
    }


def get_trimodal_specialist_logits(model, t_feat, a_feat, v_feat, out_t, out_a, out_v):
    module = get_model_module(model)

    boost_t = module.boost_head_text(t_feat) if hasattr(module, 'boost_head_text') else out_t
    boost_a = module.boost_head_audio(a_feat) if hasattr(module, 'boost_head_audio') else out_a
    boost_v = module.boost_head_visual(v_feat) if hasattr(module, 'boost_head_visual') else out_v

    specialist_t = 0.5 * out_t + 0.5 * boost_t
    specialist_a = 0.5 * out_a + 0.5 * boost_a
    specialist_v = 0.5 * out_v + 0.5 * boost_v

    return (
        {'text': specialist_t, 'audio': specialist_a, 'visual': specialist_v},
        {'text': boost_t, 'audio': boost_a, 'visual': boost_v},
    )



def ensure_trimodal_feature_repair_modules(args, model):
    if not getattr(args, 'bs_use_feature_repair', False):
        return
    module = get_model_module(model)
    if hasattr(module, 'repair_adapter_text'):
        return

    text_dim = module.head_text.in_features
    audio_dim = module.head_audio.in_features
    visual_dim = module.head_visual.in_features

    def build_adapter(in_dim):
        hidden_dim = max(in_dim // 2, 64)
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    module.repair_adapter_text = build_adapter(text_dim)
    module.repair_adapter_audio = build_adapter(audio_dim)
    module.repair_adapter_visual = build_adapter(visual_dim)


def get_trimodal_repaired_features(args, model, t_feat, a_feat, v_feat, targets):
    module = get_model_module(model)
    scale = float(getattr(args, 'bs_feature_residual_scale', 0.0))
    feature_map = {'text': t_feat, 'audio': a_feat, 'visual': v_feat}
    repaired_map = dict(feature_map)
    adapter_map = {
        'text': getattr(module, 'repair_adapter_text', None),
        'audio': getattr(module, 'repair_adapter_audio', None),
        'visual': getattr(module, 'repair_adapter_visual', None),
    }

    for target in targets:
        adapter = adapter_map.get(target, None)
        if adapter is None:
            continue
        repaired_map[target] = feature_map[target] + scale * adapter(feature_map[target])
    return repaired_map

def resolve_trimodal_specialist_targets(bs_target, conf_map):
    normalized_target = (bs_target or 'weakest').lower().replace(' ', '')
    if normalized_target in ['weakest', 'weak']:
        return [min(conf_map, key=conf_map.get)]
    if normalized_target in ['all', 'both']:
        return ['text', 'audio', 'visual']
    if normalized_target in ['non_text', 'nontext', 'av']:
        return ['audio', 'visual']

    parsed_targets = [name for name in normalized_target.split(',') if name in ['text', 'audio', 'visual']]
    if parsed_targets:
        return parsed_targets
    return [min(conf_map, key=conf_map.get)]


def apply_trimodal_specialist_eval_enhancement(args, model, t_feat, a_feat, v_feat,
                                               out_t, out_a, out_v, out_fusion):
    if not getattr(args, 'use_branch_specialist', False):
        return out_t, out_a, out_v, out_fusion

    text_blend = float(np.clip(getattr(args, 'bs_eval_text_blend', 0.0), 0.0, 1.0))
    audio_blend = float(np.clip(getattr(args, 'bs_eval_audio_blend', 0.0), 0.0, 1.0))
    visual_blend = float(np.clip(getattr(args, 'bs_eval_visual_blend', 0.0), 0.0, 1.0))
    fusion_weight = float(np.clip(getattr(args, 'bs_eval_fusion_weight', 0.0), 0.0, 1.0))

    if max(text_blend, audio_blend, visual_blend, fusion_weight) <= 0:
        return out_t, out_a, out_v, out_fusion

    conf_text = float(F.softmax(out_t.detach(), dim=1).max(dim=1)[0].mean().item())
    conf_audio = float(F.softmax(out_a.detach(), dim=1).max(dim=1)[0].mean().item())
    conf_visual = float(F.softmax(out_v.detach(), dim=1).max(dim=1)[0].mean().item())
    eval_target = getattr(args, 'bs_eval_target', '') or getattr(args, 'bs_target', 'weakest')
    targets = resolve_trimodal_specialist_targets(eval_target, {
        'text': conf_text,
        'audio': conf_audio,
        'visual': conf_visual,
    })

    specialists, _ = get_trimodal_specialist_logits(model, t_feat, a_feat, v_feat, out_t, out_a, out_v)
    base_logits = {'text': out_t, 'audio': out_a, 'visual': out_v}
    blend_map = {'text': text_blend, 'audio': audio_blend, 'visual': visual_blend}
    updated_logits = dict(base_logits)
    fusion_residuals = []

    for target in targets:
        blend = blend_map.get(target, 0.0)
        if blend <= 0:
            continue
        enhanced = (1.0 - blend) * base_logits[target] + blend * specialists[target]
        updated_logits[target] = enhanced
        fusion_residuals.append(enhanced - base_logits[target])

    if fusion_residuals and fusion_weight > 0:
        fusion_delta = torch.stack(fusion_residuals, dim=0).mean(dim=0)
        out_fusion = out_fusion + fusion_weight * fusion_delta

    return updated_logits['text'], updated_logits['audio'], updated_logits['visual'], out_fusion


def compute_trimodal_contrastive_loss(args, epoch, model, t_feat, a_feat, v_feat, contrastive_criterion):
    zero = t_feat.new_tensor(0.0)
    if not getattr(args, 'use_contrastive', False) or contrastive_criterion is None:
        return zero, None
    if epoch < getattr(args, 'con_start_epoch', 0):
        return zero, None

    module = get_model_module(model)
    if not hasattr(module, 'project_modal_features'):
        return zero, None

    proj_t, proj_a, proj_v = module.project_modal_features(t_feat, a_feat, v_feat)
    if proj_t is None or proj_a is None or proj_v is None:
        return zero, None

    loss_ta = contrastive_criterion(proj_t, proj_a)
    loss_tv = contrastive_criterion(proj_t, proj_v)
    loss_av = contrastive_criterion(proj_a, proj_v)
    loss = (loss_ta + loss_tv + loss_av) / 3.0
    return loss, {
        'con_ta': float(loss_ta.detach().item()),
        'con_tv': float(loss_tv.detach().item()),
        'con_av': float(loss_av.detach().item()),
    }


def compute_trimodal_branch_specialist_loss(args, epoch, model, t_feat, a_feat, v_feat,
                                            out_fusion, out_t, out_a, out_v, label):
    zero = out_fusion.new_tensor(0.0)
    if not getattr(args, 'use_branch_specialist', False) or epoch < args.bs_start_epoch:
        return zero, None

    conf_text, _ = compute_label_confidence(out_t, label)
    conf_audio, _ = compute_label_confidence(out_a, label)
    conf_visual, _ = compute_label_confidence(out_v, label)
    conf_map = {'text': conf_text, 'audio': conf_audio, 'visual': conf_visual}
    targets = resolve_trimodal_specialist_targets(args.bs_target, conf_map)
    specialists, boosts = get_trimodal_specialist_logits(model, t_feat, a_feat, v_feat, out_t, out_a, out_v)
    modal_logits = {'text': out_t, 'audio': out_a, 'visual': out_v}

    temperature = max(args.bs_temperature, 1e-6)
    teacher_prob = F.softmax(out_fusion.detach() / temperature, dim=1)
    teacher_conf = float(F.softmax(out_fusion.detach(), dim=1).gather(1, label.view(-1, 1)).mean().item())
    if teacher_conf < getattr(args, 'bs_min_teacher_conf', 0.0):
        return zero, {
            'bs_loss': 0.0,
            'bs_targets': ','.join(targets),
            'bs_shared_ce': 0.0,
            'bs_specialist_ce': 0.0,
            'bs_kd': 0.0,
            'bs_teacher_conf': teacher_conf,
            'bs_conf_text': conf_text,
            'bs_conf_audio': conf_audio,
            'bs_conf_visual': conf_visual,
        }

    loss = zero
    total_shared_ce = 0.0
    total_specialist_ce = 0.0
    total_kd = 0.0
    fusion_residuals = []
    for target in targets:
        shared_ce = F.cross_entropy(modal_logits[target], label)
        if getattr(args, 'bs_boost_only_ce', False):
            specialist_ce = F.cross_entropy(boosts[target], label)
        else:
            specialist_ce = 0.5 * F.cross_entropy(modal_logits[target], label) + 0.5 * F.cross_entropy(boosts[target], label)

        kd_student = specialists[target]
        if getattr(args, 'bs_detach_base_for_kd', False):
            kd_student = 0.5 * modal_logits[target].detach() + 0.5 * boosts[target]

        kd = F.kl_div(
            F.log_softmax(kd_student / temperature, dim=1),
            teacher_prob,
            reduction='batchmean'
        ) * (temperature ** 2)
        fusion_residuals.append(specialists[target] - modal_logits[target])

        loss = loss + args.bs_shared_ce_weight * shared_ce
        loss = loss + args.bs_specialist_ce_weight * specialist_ce
        loss = loss + args.bs_kd_weight * kd

        total_shared_ce += float(shared_ce.detach().item())
        total_specialist_ce += float(specialist_ce.detach().item())
        total_kd += float(kd.detach().item())

    fusion_aux_ce_value = 0.0
    if fusion_residuals and getattr(args, 'bs_fusion_aux_weight', 0.0) > 0:
        fusion_delta = torch.stack(fusion_residuals, dim=0).mean(dim=0)
        fusion_aug = out_fusion + getattr(args, 'bs_fusion_residual_scale', 0.0) * fusion_delta
        fusion_aux_ce = F.cross_entropy(fusion_aug, label)
        loss = loss + args.bs_fusion_aux_weight * fusion_aux_ce
        fusion_aux_ce_value = float(fusion_aux_ce.detach().item())

    feature_modal_ce_value = 0.0
    feature_fusion_ce_value = 0.0
    if getattr(args, 'bs_use_feature_repair', False) and (
        getattr(args, 'bs_feature_modal_ce_weight', 0.0) > 0 or getattr(args, 'bs_feature_fusion_ce_weight', 0.0) > 0
    ):
        module = get_model_module(model)
        repaired_map = get_trimodal_repaired_features(args, model, t_feat, a_feat, v_feat, targets)
        repaired_logits = {
            'text': module.head_text(repaired_map['text']),
            'audio': module.head_audio(repaired_map['audio']),
            'visual': module.head_visual(repaired_map['visual']),
        }
        if getattr(args, 'bs_feature_modal_ce_weight', 0.0) > 0:
            for target in targets:
                feature_modal_ce = F.cross_entropy(repaired_logits[target], label)
                loss = loss + args.bs_feature_modal_ce_weight * feature_modal_ce
                feature_modal_ce_value += float(feature_modal_ce.detach().item())
        if getattr(args, 'bs_feature_fusion_ce_weight', 0.0) > 0:
            repaired_fusion = module.get_fusion_output_from_features(
                repaired_map['text'], repaired_map['audio'], repaired_map['visual']
            )
            feature_fusion_ce = F.cross_entropy(repaired_fusion, label)
            loss = loss + args.bs_feature_fusion_ce_weight * feature_fusion_ce
            feature_fusion_ce_value = float(feature_fusion_ce.detach().item())

    return loss, {
        'bs_loss': float(loss.detach().item()),
        'bs_targets': ','.join(targets),
        'bs_shared_ce': total_shared_ce,
        'bs_specialist_ce': total_specialist_ce,
        'bs_kd': total_kd,
        'bs_fusion_ce': fusion_aux_ce_value,
        'bs_feature_modal_ce': feature_modal_ce_value,
        'bs_feature_fusion_ce': feature_fusion_ce_value,
        'bs_teacher_conf': teacher_conf,
        'bs_conf_text': conf_text,
        'bs_conf_audio': conf_audio,
        'bs_conf_visual': conf_visual,
    }


def compute_trimodal_single_run_repair_loss(args, epoch, model, t_feat, a_feat, v_feat,
                                            out_fusion, out_t, out_a, out_v, label):
    zero = out_fusion.new_tensor(0.0)
    if not getattr(args, 'use_single_run_repair', False):
        return zero, None
    if epoch < args.sr_start_epoch or epoch > args.sr_end_epoch:
        return zero, None

    conf_text, _ = compute_label_confidence(out_t, label)
    conf_audio, _ = compute_label_confidence(out_a, label)
    conf_visual, _ = compute_label_confidence(out_v, label)
    conf_map = {'text': conf_text, 'audio': conf_audio, 'visual': conf_visual}

    target_pool = getattr(args, 'sr_target_pool', 'all')
    if target_pool == 'non_text':
        candidate_conf_map = {'audio': conf_audio, 'visual': conf_visual}
    else:
        candidate_conf_map = conf_map

    weakest_modality = min(candidate_conf_map, key=candidate_conf_map.get)
    strongest_conf = max(candidate_conf_map.values())
    weakest_conf = candidate_conf_map[weakest_modality]
    repair_weight = float(np.clip((strongest_conf - weakest_conf) * args.sr_gap_scale, 0.0, 1.0))

    if repair_weight <= 0:
        return zero, {
            'sr_loss': 0.0,
            'sr_target': weakest_modality,
            'sr_pool': target_pool,
            'sr_weight': 0.0,
            'sr_conf_text': conf_text,
            'sr_conf_audio': conf_audio,
            'sr_conf_visual': conf_visual,
        }

    specialists, _ = get_trimodal_specialist_logits(model, t_feat, a_feat, v_feat, out_t, out_a, out_v)
    module = get_model_module(model)
    if not hasattr(module, 'get_fusion_output_from_features'):
        return zero, None

    zero_t = torch.zeros_like(t_feat)
    zero_a = torch.zeros_like(a_feat)
    zero_v = torch.zeros_like(v_feat)
    repair_inputs = {
        'text': (t_feat, zero_a, zero_v),
        'audio': (zero_t, a_feat, zero_v),
        'visual': (zero_t, zero_a, v_feat),
    }
    modal_logits = {'text': out_t, 'audio': out_a, 'visual': out_v}
    repair_fusion = module.get_fusion_output_from_features(*repair_inputs[weakest_modality])
    if getattr(args, 'sr_use_fusion_residual', False) and getattr(args, 'sr_fusion_residual_scale', 0.0) > 0:
        repair_residual = specialists[weakest_modality] - modal_logits[weakest_modality]
        repair_fusion = repair_fusion + getattr(args, 'sr_fusion_residual_scale', 0.0) * repair_residual

    temperature = max(args.sr_temperature, 1e-6)
    teacher_prob = F.softmax(out_fusion.detach() / temperature, dim=1)
    shared_ce = F.cross_entropy(modal_logits[weakest_modality], label)
    specialist_ce = F.cross_entropy(specialists[weakest_modality], label)
    fused_ce = F.cross_entropy(repair_fusion, label)
    kd = F.kl_div(
        F.log_softmax(specialists[weakest_modality] / temperature, dim=1),
        teacher_prob,
        reduction='batchmean'
    ) * (temperature ** 2)

    loss = repair_weight * (
        args.sr_ce_weight * shared_ce
        + args.sr_specialist_ce_weight * specialist_ce
        + args.sr_fused_ce_weight * fused_ce
        + args.sr_kd_weight * kd
    )
    return loss, {
        'sr_loss': float(loss.detach().item()),
        'sr_target': weakest_modality,
        'sr_pool': target_pool,
        'sr_weight': repair_weight,
        'sr_shared_ce': float(shared_ce.detach().item()),
        'sr_specialist_ce': float(specialist_ce.detach().item()),
        'sr_fused_ce': float(fused_ce.detach().item()),
        'sr_kd': float(kd.detach().item()),
        'sr_conf_text': conf_text,
        'sr_conf_audio': conf_audio,
        'sr_conf_visual': conf_visual,
    }


def train_epoch_trimodal(args, epoch, model, device, dataloader, optimizer, scheduler,
                         writer=None, global_model=None,
                         text_trace_list=None, audio_trace_list=None, visual_trace_list=None):
    """
    三模态训练一个 epoch

    新增参数:
        text_trace_list: 文本模态的 FIM trace 列表 (用于计算 k)
        audio_trace_list: 音频模态的 FIM trace 列表
        visual_trace_list: 视觉模态的 FIM trace 列表
    """
    criterion = nn.CrossEntropyLoss()
    focal_criterion = None
    if getattr(args, 'use_sample_weighting', False):
        progress = min(1.0, epoch / max(1, args.epochs - 1))
        current_gamma = args.focal_gamma_start + (args.focal_gamma_end - args.focal_gamma_start) * progress
        focal_criterion = FocalLoss(gamma=current_gamma)
    else:
        current_gamma = None
    contrastive_criterion = CrossModalContrastiveLoss() if getattr(args, 'use_contrastive', False) else None
    softmax = nn.Softmax(dim=1)

    # ==== 计算 k 值 (使用最强模态的 FIM trace 来判断 prime learning window) ====
    # 使用 10 个 epoch 的平滑来判断是否处于 prime learning window
    # 选择最强模态（FIM trace 最大的模态）
    if epoch == 0 or text_trace_list is None or len(text_trace_list) < 11:
        k = 1.0  # 初始阶段，认为处于 prime learning window
        strongest_modality = 'text'
    else:
        # 获取各模态最近的 FIM trace 平均值
        tr_text = sum(text_trace_list[-10:]) / 10
        tr_audio = sum(audio_trace_list[-10:]) / 10
        tr_visual = sum(visual_trace_list[-10:]) / 10

        # 找到最强模态 (FIM trace 最大的)
        traces = {'text': tr_text, 'audio': tr_audio, 'visual': tr_visual}
        strongest_modality = max(traces, key=traces.get)

        # 使用最强模态的 trace 来计算 k
        if strongest_modality == 'text':
            trace_list = text_trace_list
        elif strongest_modality == 'audio':
            trace_list = audio_trace_list
        else:
            trace_list = visual_trace_list

        tr1 = sum(trace_list[-10:]) / 10
        tr2 = sum(trace_list[-11:-1]) / 10

        # 计算 k: FIM 变化率
        if tr1 != 0:
            k = (tr1 - tr2) / tr1
        else:
            k = 1.0

    print("----------------------------")
    print(f"k 的值为 {k:.6f} (基于最强模态 '{strongest_modality}' 的 FIM trace)")
    in_prime_window = k > args.k_threshold if hasattr(args, 'k_threshold') else k > 0.05
    print(f"Prime Learning Window: {'是' if in_prime_window else '否'} (阈值: {getattr(args, 'k_threshold', 0.05)})")
    print("----------------------------")

    if global_model is None:
        global_model = cp.deepcopy(model)

    model.train()
    print(f"Start training epoch {epoch}...")

    # 记录参数名
    record_names_text = []
    record_names_audio = []
    record_names_visual = []

    for name, param in model.named_parameters():
        if 'head' in name and 'fusion' not in name:
            continue  # 跳过单模态 head
        if 'text' in name:
            record_names_text.append((name, param))
        elif 'audio' in name:
            record_names_audio.append((name, param))
        elif 'visual' in name:
            record_names_visual.append((name, param))

    # ==== 初始化 FIM 字典 (用于计算各模态的 Fisher 信息) ====
    fim_text = {}
    fim_audio = {}
    fim_visual = {}

    # 获取实际的模型 (处理 DataParallel)
    actual_model = model.module if hasattr(model, 'module') else model

    for name, module in actual_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            if module.weight.requires_grad:
                # 根据名称分配到不同模态
                if 'text' in name:
                    fim_text[name] = torch.zeros_like(module.weight)
                elif 'audio' in name:
                    fim_audio[name] = torch.zeros_like(module.weight)
                elif 'visual' in name:
                    fim_visual[name] = torch.zeros_like(module.weight)

    total_loss = 0
    total_loss_t = 0
    total_loss_a = 0
    total_loss_v = 0
    total_correct = 0
    total_samples = 0
    total_loss_con = 0.0
    total_loss_bs = 0.0
    total_loss_sr = 0.0
    total_loss_phsf_aux = 0.0
    total_weight_text = 0.0
    total_weight_audio = 0.0
    total_weight_visual = 0.0

    for step, (text, audio, visual, label) in enumerate(dataloader):
        text = text.to(device).float()
        audio = audio.to(device).float()
        visual = visual.to(device).float()
        label = label.to(device).long()

        batch_size = label.shape[0]
        total_samples += batch_size

        optimizer.zero_grad()

        # Forward pass
        t_feat, a_feat, v_feat, out_t, out_a, out_v, out_fusion = model(text, audio, visual)

        # 计算各模态 loss
        focal_scope = getattr(args, 'sample_weighting_scope', 'all')
        fusion_focal = focal_criterion if focal_scope in ['all', 'fusion_only'] else None
        unimodal_focal = focal_criterion if focal_scope in ['all', 'unimodal_only'] else None

        loss_fusion = compute_cls_loss(out_fusion, label, criterion, fusion_focal)
        loss_t = compute_cls_loss(out_t, label, criterion, unimodal_focal)
        loss_a = compute_cls_loss(out_a, label, criterion, unimodal_focal)
        loss_v = compute_cls_loss(out_v, label, criterion, unimodal_focal)

        # 计算各模态的 score (softmax 概率在真实标签处的和)
        score_t = softmax(out_t).gather(1, label.view(-1, 1)).sum()
        score_a = softmax(out_a).gather(1, label.view(-1, 1)).sum()
        score_v = softmax(out_v).gather(1, label.view(-1, 1)).sum()

        # 计算三模态的 beta
        beta_t, beta_a, beta_v = compute_trimodal_beta(
            score_t, score_a, score_v, alpha=args.beta_scale
        )

        # CMoB 作为可选增强项独立叠加；不开时保持原始 baseline 损失不变。
        if getattr(args, 'use_cmob', False):
            causal_weights, causal_stats = compute_trimodal_causal_weights(
                model, t_feat, a_feat, v_feat, out_fusion, label, criterion
            )
            unimodal_loss = (
                causal_weights['text'] * loss_t
                + causal_weights['audio'] * loss_a
                + causal_weights['visual'] * loss_v
            ) / 3.0
            total_weight_text += causal_weights['text']
            total_weight_audio += causal_weights['audio']
            total_weight_visual += causal_weights['visual']
            total_loss_item = loss_fusion + args.cmob_weight * unimodal_loss
        elif args.trimodal_loss_protocol == 'paper':
            causal_weights = {'text': 1.0, 'audio': 1.0, 'visual': 1.0}
            causal_stats = None
            total_loss_item = loss_fusion
        else:
            causal_weights = {'text': 1.0, 'audio': 1.0, 'visual': 1.0}
            causal_stats = None
            total_loss_item = loss_fusion + loss_t + loss_a + loss_v

        phsf_aux_loss = out_fusion.new_tensor(0.0)
        phsf_aux_weight = float(getattr(args, 'phsf_aux_weight', 0.0))
        phsf_aux_enabled = (
            getattr(args, 'use_sample_weighting', False)
            and phsf_aux_weight > 0
            and epoch >= int(getattr(args, 'phsf_aux_start_epoch', 0))
            and (not getattr(args, 'use_cmob', False) or getattr(args, 'phsf_aux_include_cmdr', False))
        )
        if phsf_aux_enabled:
            if getattr(args, 'phsf_aux_targets', 'non_text') == 'all':
                phsf_aux_loss = (loss_t + loss_a + loss_v) / 3.0
            else:
                phsf_aux_loss = (loss_a + loss_v) / 2.0
            total_loss_item = total_loss_item + phsf_aux_weight * phsf_aux_loss
            total_loss_phsf_aux += float(phsf_aux_loss.detach().item())

        contrastive_loss, contrastive_stats = compute_trimodal_contrastive_loss(
            args, epoch, model, t_feat, a_feat, v_feat, contrastive_criterion
        )
        if contrastive_stats is not None:
            total_loss_item = total_loss_item + args.con_weight * contrastive_loss
            total_loss_con += float(contrastive_loss.detach().item())
        else:
            contrastive_loss = out_fusion.new_tensor(0.0)

        specialist_loss, specialist_stats = compute_trimodal_branch_specialist_loss(
            args, epoch, model, t_feat, a_feat, v_feat, out_fusion, out_t, out_a, out_v, label
        )
        total_loss_item = total_loss_item + specialist_loss
        total_loss_bs += float(specialist_loss.detach().item())

        repair_loss, repair_stats = compute_trimodal_single_run_repair_loss(
            args, epoch, model, t_feat, a_feat, v_feat, out_fusion, out_t, out_a, out_v, label
        )
        total_loss_item = total_loss_item + repair_loss
        total_loss_sr += float(repair_loss.detach().item())

        # Backward
        total_loss_item.backward()

        # 应用 Proximal 正则化 (根据 beta 调整梯度)
        # 只有当 k > k_threshold 时才应用，表示模型处于 prime learning window
        if args.modulation == 'Ours' and k > args.k_threshold:
            for model_param, global_param in zip(model.parameters(), global_model.parameters()):
                if model_param.requires_grad and model_param.grad is not None:
                    # 检查参数属于哪个模态
                    param_name = None
                    for name, p in model.named_parameters():
                        if p is model_param:
                            param_name = name
                            break

                    if param_name is not None:
                        # 支持两种命名: text_net/audio_net/visual_net 或 text_encoder/audio_encoder/visual_encoder
                        if 'text_net' in param_name or 'text_encoder' in param_name:
                            model_param.grad += beta_t * (model_param - global_param)
                        elif 'audio_net' in param_name or 'audio_encoder' in param_name:
                            model_param.grad += beta_a * (model_param - global_param)
                        elif 'visual_net' in param_name or 'visual_encoder' in param_name:
                            model_param.grad += beta_v * (model_param - global_param)

        # ==== 累积 FIM (Fisher Information Matrix) ====
        # FIM ≈ E[∇log p(y|x) * ∇log p(y|x)^T] ≈ 梯度的平方
        for name, module in actual_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if module.weight.requires_grad and module.weight.grad is not None:
                    grad_sq = module.weight.grad.data ** 2
                    if 'text' in name and name in fim_text:
                        fim_text[name] += grad_sq
                    elif 'audio' in name and name in fim_audio:
                        fim_audio[name] += grad_sq
                    elif 'visual' in name and name in fim_visual:
                        fim_visual[name] += grad_sq

        optimizer.step()
        scheduler.step()  # Cosine scheduler: step after each batch

        # 统计
        total_loss += total_loss_item.item()
        total_loss_t += loss_t.item()
        total_loss_a += loss_a.item()
        total_loss_v += loss_v.item()

        _, predicted = torch.max(out_fusion, 1)
        total_correct += (predicted == label).sum().item()

        # 打印进度
        if step % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            # 判断是否应用了 modulation
            modulation_applied = args.modulation == 'Ours' and k > args.k_threshold
            mod_status = "[MOD ON]" if modulation_applied else "[MOD OFF]"
            print(f"  Step {step}/{len(dataloader)}, "
                  f"Loss: {total_loss_item.item():.4f}, LR: {current_lr:.6f}, "
                  f"beta_t: {beta_t.item():.4f}, beta_a: {beta_a.item():.4f}, beta_v: {beta_v.item():.4f}, "
                  f"phsf_aux: {phsf_aux_loss.item():.4f}, con: {contrastive_loss.item():.4f}, "
                  f"bs: {specialist_loss.item():.4f}, sr: {repair_loss.item():.4f}, "
                  f"protocol: {args.trimodal_loss_protocol} {mod_status}")

    # ==== 计算各模态的 FIM Trace ====
    fim_trace_text = 0
    for name in fim_text:
        fim_text[name] = fim_text[name].mean().item()
        fim_trace_text += fim_text[name]

    fim_trace_audio = 0
    for name in fim_audio:
        fim_audio[name] = fim_audio[name].mean().item()
        fim_trace_audio += fim_audio[name]

    fim_trace_visual = 0
    for name in fim_visual:
        fim_visual[name] = fim_visual[name].mean().item()
        fim_trace_visual += fim_visual[name]

    # 将 FIM trace 添加到列表中 (用于下一个 epoch 计算 k)
    if text_trace_list is not None:
        text_trace_list.append(fim_trace_text)
    if audio_trace_list is not None:
        audio_trace_list.append(fim_trace_audio)
    if visual_trace_list is not None:
        visual_trace_list.append(fim_trace_visual)

    print(f"FIM Trace - Text: {fim_trace_text:.6f}, Audio: {fim_trace_audio:.6f}, Visual: {fim_trace_visual:.6f}")

    # Note: scheduler.step() 已在每个 batch 后调用 (cosine scheduler)

    # 计算平均值
    avg_loss = total_loss / len(dataloader)
    avg_loss_t = total_loss_t / len(dataloader)
    avg_loss_a = total_loss_a / len(dataloader)
    avg_loss_v = total_loss_v / len(dataloader)
    accuracy = total_correct / total_samples

    # TensorBoard 记录
    current_lr = optimizer.param_groups[0]['lr']
    if writer is not None:
        writer.add_scalar('Loss/fusion', avg_loss, epoch)
        writer.add_scalar('Loss/text', avg_loss_t, epoch)
        writer.add_scalar('Loss/audio', avg_loss_a, epoch)
        writer.add_scalar('Loss/visual', avg_loss_v, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        writer.add_scalar('FIM_Trace/text', fim_trace_text, epoch)
        writer.add_scalar('FIM_Trace/audio', fim_trace_audio, epoch)
        writer.add_scalar('FIM_Trace/visual', fim_trace_visual, epoch)
        writer.add_scalar('k_value', k, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        writer.add_scalar('Loss/contrastive', total_loss_con / len(dataloader), epoch)
        writer.add_scalar('Loss/branch_specialist', total_loss_bs / len(dataloader), epoch)
        writer.add_scalar('Loss/single_run_repair', total_loss_sr / len(dataloader), epoch)
        if getattr(args, 'use_cmob', False):
            writer.add_scalar('CMoB/weight_text', total_weight_text / len(dataloader), epoch)
            writer.add_scalar('CMoB/weight_audio', total_weight_audio / len(dataloader), epoch)
            writer.add_scalar('CMoB/weight_visual', total_weight_visual / len(dataloader), epoch)
        if current_gamma is not None:
            writer.add_scalar('SampleWeighting/focal_gamma', current_gamma, epoch)

    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, k={k:.6f}")

    return avg_loss, avg_loss_t, avg_loss_a, avg_loss_v, accuracy, fim_trace_text, fim_trace_audio, fim_trace_visual, k


def valid_trimodal(args, model, device, dataloader):
    """
    验证/测试函数
    """
    n_classes = args.n_classes

    with torch.no_grad():
        model.eval()

        labels_all = []
        pred_fusion_all = []
        pred_t_all = []
        pred_a_all = []
        pred_v_all = []

        for step, (text, audio, visual, label) in enumerate(dataloader):
            text = text.to(device).float()
            audio = audio.to(device).float()
            visual = visual.to(device).float()
            label = label.to(device).long()

            t_feat, a_feat, v_feat, out_t, out_a, out_v, out_fusion = model(text, audio, visual)
            out_t, out_a, out_v, out_fusion = apply_trimodal_specialist_eval_enhancement(
                args, model, t_feat, a_feat, v_feat, out_t, out_a, out_v, out_fusion
            )

            labels_all.extend(label.cpu().numpy().tolist())
            pred_fusion_all.extend(out_fusion.argmax(dim=1).cpu().numpy().tolist())
            pred_t_all.extend(out_t.argmax(dim=1).cpu().numpy().tolist())
            pred_a_all.extend(out_a.argmax(dim=1).cpu().numpy().tolist())
            pred_v_all.extend(out_v.argmax(dim=1).cpu().numpy().tolist())

    metrics = {
        'fusion': summarize_classification_metrics(labels_all, pred_fusion_all, n_classes),
        'text': summarize_classification_metrics(labels_all, pred_t_all, n_classes),
        'audio': summarize_classification_metrics(labels_all, pred_a_all, n_classes),
        'visual': summarize_classification_metrics(labels_all, pred_v_all, n_classes),
    }

    return (
        resolve_trimodal_primary_metric(args, metrics['fusion']),
        resolve_trimodal_primary_metric(args, metrics['text']),
        resolve_trimodal_primary_metric(args, metrics['audio']),
        resolve_trimodal_primary_metric(args, metrics['visual']),
        metrics,
    )


def apply_mosi_adaptation_defaults(args):
    """
    MOSI ????????????????????????????
    ? branch-specialist ???????????
    ???????????
    """
    if args.dataset != 'MOSI':
        return args

    # ???1??? CMoB ??????????
    if getattr(args, 'use_cmob', False):
        args.cmob_weight = min(args.cmob_weight, args.mosi_adapt_cmob_cap)

    # ???2???????????????? Focal ???
    # ?????????? unimodal_only??????? fusion ???
    if getattr(args, 'use_sample_weighting', False):
        args.sample_weighting_scope = 'unimodal_only'
        args.focal_gamma_start = min(args.focal_gamma_start, args.mosi_adapt_focal_start_cap)
        args.focal_gamma_end = min(args.focal_gamma_end, args.mosi_adapt_focal_end_cap)

    print('[MOSI-ADAPT] Applied conservative settings for CMoB + SW:')
    print(f"  cmob_weight={getattr(args, 'cmob_weight', None)}")
    print(f"  sample_weighting_scope={getattr(args, 'sample_weighting_scope', None)}")
    print(f"  focal_gamma_start={getattr(args, 'focal_gamma_start', None)}")
    print(f"  focal_gamma_end={getattr(args, 'focal_gamma_end', None)}")
    print(f"  phsf_aux_weight={getattr(args, 'phsf_aux_weight', None)}")
    print(f"  phsf_aux_targets={getattr(args, 'phsf_aux_targets', None)}")
    print(f"  phsf_aux_include_cmdr={getattr(args, 'phsf_aux_include_cmdr', None)}")
    print(f"  mosi_adapt_cmob_cap={getattr(args, 'mosi_adapt_cmob_cap', None)}")
    print(f"  mosi_adapt_focal_end_cap={getattr(args, 'mosi_adapt_focal_end_cap', None)}")
    return args


def main():
    args = get_arguments()
    args = resolve_paper_like_trimodal_chain(args)
    args = apply_mosi_adaptation_defaults(args)
    print("=" * 60)
    print("三模态训练 - MOSEI/MOSI")
    print("=" * 60)
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, GPU ids: {gpu_ids}")

    # 数据集 (先加载数据集，因为 scheduler 需要 len(train_loader))
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_mosei_dataloaders(
        args, batch_size=args.batch_size, num_workers=args.data_workers
    )
    args = infer_trimodal_feature_dims(train_loader, args)

    # 创建模型 - 根据 model_type 选择
    if args.model_type == 'transformer':
        print("Using TriModalTransformer (Text-Transformer, Audio/Visual-MLP)")
        model = TriModalTransformer(args)
    else:
        print("Using TriModalClassifier (Simple MLP for all modalities)")
        model = TriModalClassifier(args)

    ensure_trimodal_feature_repair_modules(args, model)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    model.to(device)

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # 优化器 (参考 mosei: SGD, lr=0.001, momentum=0.9, weight_decay=1e-4)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                               betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)

    # Scheduler (参考 mosei: cosine_schedule_with_warmup)
    # warmup_steps = len(train_loader) * warmup_epochs
    # total_steps = len(train_loader) * n_epochs
    num_warmup_steps = int(len(train_loader) * args.warmup_epochs)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"Using CosineScheduleWithWarmup: warmup_steps={num_warmup_steps}, total_steps={num_training_steps}")
    
    if args.train:
        # TensorBoard
        writer = None
        if args.use_tensorboard:
            writer_path = os.path.join(args.tensorboard_path, args.dataset)
            os.makedirs(writer_path, exist_ok=True)
            log_name = f'{args.modulation}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            writer = SummaryWriter(os.path.join(writer_path, log_name))
        
        best_acc = 0.0
        best_ckpt_path = None
        global_model = cp.deepcopy(model)

        # 记录列表
        loss_list = []
        acc_list = []
        acc_t_list = []
        acc_a_list = []
        acc_v_list = []

        # ==== FIM Trace 列表 (初始化 10 个 0 做平滑) ====
        # 用于判断 prime learning window
        text_trace_list = [0.0] * 10
        audio_trace_list = [0.0] * 10
        visual_trace_list = [0.0] * 10

        # k 值列表 (记录学习窗口状态)
        k_list = []

        for epoch in range(args.epochs):
            print(f'\n===== Epoch {epoch} =====')

            # 训练
            (avg_loss, loss_t, loss_a, loss_v, train_acc,
             fim_trace_text, fim_trace_audio, fim_trace_visual, k) = train_epoch_trimodal(
                args, epoch, model, device, dataloader=train_loader,
                optimizer=optimizer, scheduler=scheduler, writer=writer,
                global_model=global_model,
                text_trace_list=text_trace_list,
                audio_trace_list=audio_trace_list,
                visual_trace_list=visual_trace_list
            )

            loss_list.append(avg_loss)
            k_list.append(k)

            # 验证
            acc, acc_t, acc_a, acc_v, val_metrics = valid_trimodal(args, model, device, val_loader)
            acc_list.append(acc)
            acc_t_list.append(acc_t)
            acc_a_list.append(acc_a)
            acc_v_list.append(acc_v)

            if writer is not None:
                writer.add_scalar('Accuracy/val_fusion', acc, epoch)
                writer.add_scalar('Accuracy/val_text', acc_t, epoch)
                writer.add_scalar('Accuracy/val_audio', acc_a, epoch)
                writer.add_scalar('Accuracy/val_visual', acc_v, epoch)
                writer.add_scalar('AccuracyOverall/val_fusion', val_metrics['fusion']['overall_acc'], epoch)
                writer.add_scalar('AccuracyOverall/val_text', val_metrics['text']['overall_acc'], epoch)
                writer.add_scalar('AccuracyOverall/val_audio', val_metrics['audio']['overall_acc'], epoch)
                writer.add_scalar('AccuracyOverall/val_visual', val_metrics['visual']['overall_acc'], epoch)
                writer.add_scalar('AccuracyPaper/val_fusion', val_metrics['fusion']['paper_acc'], epoch)
                writer.add_scalar('AccuracyPaper/val_text', val_metrics['text']['paper_acc'], epoch)
                writer.add_scalar('AccuracyPaper/val_audio', val_metrics['audio']['paper_acc'], epoch)
                writer.add_scalar('AccuracyPaper/val_visual', val_metrics['visual']['paper_acc'], epoch)
                writer.add_scalar('MacroF1/val_fusion', val_metrics['fusion']['macro_f1'], epoch)
                writer.add_scalar('MacroF1/val_text', val_metrics['text']['macro_f1'], epoch)
                writer.add_scalar('MacroF1/val_audio', val_metrics['audio']['macro_f1'], epoch)
                writer.add_scalar('MacroF1/val_visual', val_metrics['visual']['macro_f1'], epoch)

            print(
                "Val - "
                f"Fusion(PaperAcc={val_metrics['fusion']['paper_acc']:.4f}, OverallAcc={val_metrics['fusion']['overall_acc']:.4f}, MacroF1={val_metrics['fusion']['macro_f1']:.4f}), "
                f"Text(PaperAcc={val_metrics['text']['paper_acc']:.4f}, OverallAcc={val_metrics['text']['overall_acc']:.4f}, MacroF1={val_metrics['text']['macro_f1']:.4f}), "
                f"Audio(PaperAcc={val_metrics['audio']['paper_acc']:.4f}, OverallAcc={val_metrics['audio']['overall_acc']:.4f}, MacroF1={val_metrics['audio']['macro_f1']:.4f}), "
                f"Visual(PaperAcc={val_metrics['visual']['paper_acc']:.4f}, OverallAcc={val_metrics['visual']['overall_acc']:.4f}, MacroF1={val_metrics['visual']['macro_f1']:.4f})"
            )

            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                os.makedirs(args.ckpt_path, exist_ok=True)
                
                model_name = f'best_model_{args.dataset}_{args.modulation}_epoch{epoch}_acc{acc:.4f}.pth'
                save_path = os.path.join(args.ckpt_path, model_name)
                
                saved_dict = {
                    'epoch': epoch,
                    'modulation': args.modulation,
                    'acc': acc,
                    'model': get_model_module(model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(saved_dict, save_path)
                print(f'Best model saved to {save_path}')
                best_ckpt_path = save_path
            
            # 更新 global_model (每 epoch 更新一次)
            global_model = cp.deepcopy(model)

        # 最终测试前，先回载 best val 模型，保证评测口径符合学术常规
        if best_ckpt_path is not None:
            loaded_dict = torch.load(best_ckpt_path, map_location=device)
            get_model_module(model).load_state_dict(loaded_dict['model'])
            print(f'Best-val model reloaded from {best_ckpt_path}')

        # 最终测试
        test_acc, test_acc_t, test_acc_a, test_acc_v, test_metrics = valid_trimodal(args, model, device, test_loader)
        print(f"\n===== Final Test Results =====")
        print(f"PrimaryMetric - Fusion: {test_acc:.4f}, Text: {test_acc_t:.4f}, Audio: {test_acc_a:.4f}, Visual: {test_acc_v:.4f}")
        print(
            "Overall - "
            f"Fusion(Acc={test_metrics['fusion']['overall_acc']:.4f}), "
            f"Text(Acc={test_metrics['text']['overall_acc']:.4f}), "
            f"Audio(Acc={test_metrics['audio']['overall_acc']:.4f}), "
            f"Visual(Acc={test_metrics['visual']['overall_acc']:.4f})"
        )
        print(
            "Paper Compare - "
            f"Fusion(Acc={test_metrics['fusion']['paper_acc']:.4f}, MacroF1={test_metrics['fusion']['macro_f1']:.4f}), "
            f"Text(Acc={test_metrics['text']['paper_acc']:.4f}, MacroF1={test_metrics['text']['macro_f1']:.4f}), "
            f"Audio(Acc={test_metrics['audio']['paper_acc']:.4f}, MacroF1={test_metrics['audio']['macro_f1']:.4f}), "
            f"Visual(Acc={test_metrics['visual']['paper_acc']:.4f}, MacroF1={test_metrics['visual']['macro_f1']:.4f})"
        )
        print(f"Best Val Acc: {best_acc:.4f}")

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join('results', f'trimodal_{args.dataset}_{timestamp}')
        os.makedirs(results_path, exist_ok=True)

        data_to_save = {
            'loss_list': loss_list,
            'acc_list': acc_list,
            'acc_t_list': acc_t_list,
            'acc_a_list': acc_a_list,
            'acc_v_list': acc_v_list,
            'text_trace_list': text_trace_list,
            'audio_trace_list': audio_trace_list,
            'visual_trace_list': visual_trace_list,
            'k_list': k_list,
        }

        for name, data in data_to_save.items():
            pkl_path = os.path.join(results_path, f'{name}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)
        
        print(f"Results saved to {results_path}")
        if writer is not None:
            writer.close()
    
    else:
        # 测试模式
        eval_ckpt_path = resolve_eval_checkpoint_path(args)
        if eval_ckpt_path is None:
            print(f'No checkpoint found under {args.ckpt_path}')
            return

        loaded_dict = torch.load(eval_ckpt_path, map_location=device)
        get_model_module(model).load_state_dict(loaded_dict['model'])
        print(f'Model loaded from {eval_ckpt_path}')
        
        test_acc, test_acc_t, test_acc_a, test_acc_v, test_metrics = valid_trimodal(args, model, device, test_loader)
        print(f"Test - PrimaryMetric Fusion: {test_acc:.4f}, Text: {test_acc_t:.4f}, Audio: {test_acc_a:.4f}, Visual: {test_acc_v:.4f}")
        print(
            "Overall - "
            f"Fusion(Acc={test_metrics['fusion']['overall_acc']:.4f}), "
            f"Text(Acc={test_metrics['text']['overall_acc']:.4f}), "
            f"Audio(Acc={test_metrics['audio']['overall_acc']:.4f}), "
            f"Visual(Acc={test_metrics['visual']['overall_acc']:.4f})"
        )
        print(
            "Paper Compare - "
            f"Fusion(Acc={test_metrics['fusion']['paper_acc']:.4f}, MacroF1={test_metrics['fusion']['macro_f1']:.4f}), "
            f"Text(Acc={test_metrics['text']['paper_acc']:.4f}, MacroF1={test_metrics['text']['macro_f1']:.4f}), "
            f"Audio(Acc={test_metrics['audio']['paper_acc']:.4f}, MacroF1={test_metrics['audio']['macro_f1']:.4f}), "
            f"Visual(Acc={test_metrics['visual']['paper_acc']:.4f}, MacroF1={test_metrics['visual']['macro_f1']:.4f})"
        )


if __name__ == "__main__":
    main()
