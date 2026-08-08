import argparse
# 666
import os
import copy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb
import torch.nn.functional as F
from dataset.CramedDataset import CramedDataset
from utils.cmob import CMoB
from models.contrastive_loss import CrossModalContrastiveLoss
from models.focal_loss import FocalLoss
from dataset.dataset import AVDataset
from models.basic_model import *
from utils.utils import setup_seed, weight_init
from utils.triplet_goal import (
    evaluate_triplet_goal,
    is_better_triplet_goal,
    print_triplet_goal_status,
    resolve_triplet_goal_targets,
    save_goal_checkpoint,
)
from utils.weak_modality_balance import (
    branch_specialist_is_active,
    get_specialist_logits,
    resolve_single_run_repair_weights,
    single_run_repair_is_active,
)
import torchvision
import os
import pickle
import csv
import subprocess
import sys
from datetime import datetime
from KSDataset import *
import matplotlib.pyplot as plt
import numpy as np
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='Ours', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE',"Ours"])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=3, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    # ===== KS 相对 baseline（3）的关键新增 start: 多视角评估开关 =====
    # baseline（3）没有这组参数；这里把 KS 的测试阶段做成可控的 multi-view eval。
    parser.add_argument('--ks_eval_num_views', default=1, type=int,
                        help='Number of deterministic visual TTA views used for KineticSound evaluation')
    parser.add_argument('--ks_eval_resize', default=256, type=int,
                        help='Resize size used before deterministic KineticSound evaluation crops')
    # ===== KS 相对 baseline（3）的关键新增 end: 多视角评估开关 =====
    parser.add_argument('--audio_path', default='/data/Lab105/Datasets/CREMA-D/CREMAD/AudioWAV/', type=str)
    parser.add_argument('--visual_path', default='/data/Lab105/Datasets/CREMA-D/CREMAD/', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=30, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', default=0.8, type=float, help='alpha in OGM-GE')
    
    parser.add_argument('--ckpt_path', default= r'./ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=True, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='./logs',type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--audio_fim_path',default='./audio_fim_folder',type=str,help='path to store the audio fim')
    parser.add_argument('--visual_fim_path',default='./visual_fim_folder',type=str,help='path to store the visual fim')
    parser.add_argument('--mm_fim_path',default='./mm_fim_folder',type=str,help='path to store the mm fim')
    parser.add_argument('--accuracy_path',default='./accuracy_folder',type=str,help='path to store accuracy csv file')
    parser.add_argument('--use_cmob', action='store_true', help='Use Causal Modality Valuation for balancing')
    parser.add_argument('--cmdr_beta_scale', default=0.95, type=float,
                        help='Base scale for CMDR gradient modulation; default reproduces previous behavior')
    parser.add_argument('--use_contrastive', action='store_true', help='Use Cross-Modal Contrastive Loss for feature alignment')
    parser.add_argument('--con_weight', default=0.1, type=float, help='Weight for contrastive loss')
    parser.add_argument('--use_sample_weighting', action='store_true', help='Use Sample-Level Weighting (Focal Loss)')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Gamma for Focal Loss (deprecated, use start/end)')
    parser.add_argument('--focal_gamma_start', default=0.0, type=float, help='Starting gamma for Focal Loss')
    parser.add_argument('--focal_gamma_end', default=2.0, type=float, help='Ending gamma for Focal Loss')
    parser.add_argument('--sw_mode', default='prime_window', type=str, choices=['prime_window', 'always', 'epoch_range'])
    parser.add_argument('--sw_k_threshold', default=0.05, type=float)
    parser.add_argument('--sw_start_epoch', default=0, type=int)
    parser.add_argument('--sw_end_epoch', default=30, type=int)
    parser.add_argument('--use_dual_rescue', action='store_true',
                        help='Use dual-window unimodal rescue with auxiliary branch supervision and distillation')
    parser.add_argument('--rescue_ce_weight', default=0.2, type=float,
                        help='Weight of auxiliary unimodal CE loss in dual rescue')
    parser.add_argument('--rescue_kd_weight', default=0.05, type=float,
                        help='Weight of fused-to-unimodal distillation loss in dual rescue')
    parser.add_argument('--rescue_temperature', default=2.0, type=float,
                        help='Temperature for fused-to-unimodal distillation')
    parser.add_argument('--rescue_progress_weight', default=0.5, type=float,
                        help='How much modality progress lag contributes to rescue strength')
    parser.add_argument('--rescue_margin', default=0.02, type=float,
                        help='Confidence margin before activating rescue on a lagging modality')
    parser.add_argument('--rescue_eval_blend', default=0.0, type=float,
                        help='Blend auxiliary unimodal heads into unimodal evaluation logits; 0 keeps original metric')
    parser.add_argument('--use_branch_calibration', action='store_true',
                        help='Late-stage unimodal branch calibration on detached features')
    parser.add_argument('--bc_start_epoch', default=30, type=int,
                        help='Start epoch for branch calibration')
    parser.add_argument('--bc_end_epoch', default=49, type=int,
                        help='End epoch for branch calibration')
    parser.add_argument('--bc_target', default='auto', type=str,
                        choices=['auto', 'audio', 'visual', 'both'],
                        help='Which modality branch to calibrate')
    parser.add_argument('--bc_ce_weight', default=0.4, type=float,
                        help='Weight of branch CE calibration loss')
    parser.add_argument('--bc_kd_weight', default=0.1, type=float,
                        help='Weight of branch KD calibration loss')
    parser.add_argument('--bc_temperature', default=2.0, type=float,
                        help='Temperature for branch calibration distillation')
    parser.add_argument('--bc_auto_margin', default=0.02, type=float,
                        help='Margin for activating auto branch calibration')
    parser.add_argument('--bc_audio_scale', default=1.0, type=float,
                        help='Relative strength for audio branch calibration')
    parser.add_argument('--bc_visual_scale', default=1.0, type=float,
                        help='Relative strength for visual branch calibration')
    parser.add_argument('--bc_blend_audio', default=0.0, type=float,
                        help='Blend ratio of auxiliary audio head into unimodal audio evaluation')
    parser.add_argument('--bc_blend_visual', default=0.0, type=float,
                        help='Blend ratio of auxiliary visual head into unimodal visual evaluation')
    parser.add_argument('--bc_sigma', default=1.0, type=float,
                        help='RML-style confidence ratio threshold for weak modality selection')
    parser.add_argument('--bc_tau', default=0.02, type=float,
                        help='RML-style dead-zone threshold for adaptive branch boosting')
    parser.add_argument('--bc_aux_scale', default=0.5, type=float,
                        help='Scale for the auxiliary unimodal classifier in branch boosting')
    parser.add_argument('--bc_residual_scale', default=0.5, type=float,
                        help='Scale for the residual boosting classifier in branch boosting')
    parser.add_argument('--use_lrmo', action='store_true',
                        help='Use LRMO-lite adaptive subnetwork masking for modality rebalancing')
    parser.add_argument('--lrmo_start_epoch', default=5, type=int,
                        help='Start epoch for LRMO-lite gradient masking')
    parser.add_argument('--lrmo_end_epoch', default=45, type=int,
                        help='End epoch for LRMO-lite gradient masking')
    parser.add_argument('--lrmo_tau', default=0.5, type=float,
                        help='Temperature controlling modality update-ratio disparity in LRMO-lite')
    parser.add_argument('--lrmo_ema', default=0.9, type=float,
                        help='EMA factor for modality significance estimates in LRMO-lite')
    parser.add_argument('--lrmo_ratio_floor', default=0.35, type=float,
                        help='Minimum keep ratio for any modality in LRMO-lite')
    parser.add_argument('--lrmo_ratio_ceiling', default=0.95, type=float,
                        help='Maximum keep ratio for any modality in LRMO-lite')
    parser.add_argument('--lrmo_importance_source', default='grad', type=str,
                        choices=['grad', 'fim'],
                        help='Importance source for LRMO-lite mask sampling')
    parser.add_argument('--lrmo_unbiased', action='store_true',
                        help='Use AMSS+-style inverse keep-ratio scaling on sampled gradients')
    parser.add_argument('--lrmo_scale_cap', default=2.5, type=float,
                        help='Maximum scaling factor used by unbiased LRMO-lite masking')
    parser.add_argument('--lrmo_include_heads', action='store_true',
                        help='Also apply LRMO-lite masking to unimodal classifier heads')
    parser.add_argument('--use_mag', action='store_true',
                        help='Use pluggable MAG-lite feature gating in the backbone head')
    parser.add_argument('--mag_hidden_dim', default=128, type=int,
                        help='Hidden dimension of the MAG-lite gate MLP')
    parser.add_argument('--mag_dropout', default=0.1, type=float,
                        help='Dropout ratio used in the MAG-lite gate MLP')
    parser.add_argument('--mag_temperature', default=1.0, type=float,
                        help='Softmax temperature used by MAG-lite gating')
    parser.add_argument('--mag_feature_scale', default=0.4, type=float,
                        help='Residual feature scaling strength used by MAG-lite gating')
    parser.add_argument('--mag_balance_weight', default=0.0, type=float,
                        help='Weight of a light gate-balance regularizer to avoid gate collapse')
    parser.add_argument('--mag_balance_start_epoch', default=5, type=int,
                        help='Start epoch for the MAG-lite gate-balance regularizer')
    parser.add_argument('--mag_compensation_weight', default=0.0, type=float,
                        help='Weight of weak-modality compensation supervision for MAG-lite')
    parser.add_argument('--mag_compensation_start_epoch', default=8, type=int,
                        help='Start epoch for weak-modality compensation supervision')
    parser.add_argument('--mag_compensation_margin', default=0.02, type=float,
                        help='Confidence margin before MAG-lite starts favoring the weaker modality')
    parser.add_argument('--mag_compensation_temperature', default=0.25, type=float,
                        help='Temperature used when converting inverse unimodal confidence into MAG targets')
    # ===== KS 相对 baseline（3）的关键新增 start: branch specialist 开关 =====
    # 这组开关主要用于“保留 baseline 主干训练不动，只轻量补单模态分支”，
    # 其中 KS 当前成功版主要使用 audio specialist 去尽量拉总 Acc，同时少破坏 Visual Acc。
    parser.add_argument('--use_branch_specialist', action='store_true',
                        help='Train unimodal specialist heads and blend them into unimodal evaluation')
    parser.add_argument('--wbsr_loss_scale', default=1.0, type=float,
                        help='Global scale for WBSR branch-specialist loss; default reproduces previous behavior')
    parser.add_argument('--bs_start_epoch', default=6, type=int,
                        help='Start epoch for branch-specialist optimization and evaluation blending')
    parser.add_argument('--bs_target', default='both', type=str,
                        choices=['audio', 'visual', 'both'],
                        help='Target modality for branch-specialist optimization')
    parser.add_argument('--bs_audio_ce_weight', default=0.0, type=float,
                        help='Extra CE weight for the audio shared-logit path during branch-specialist tuning')
    parser.add_argument('--bs_visual_ce_weight', default=0.0, type=float,
                        help='Extra CE weight for the visual shared-logit path during branch-specialist tuning')
    parser.add_argument('--bs_audio_specialist_ce_weight', default=0.4, type=float,
                        help='CE weight for the audio specialist logits during branch-specialist tuning')
    parser.add_argument('--bs_visual_specialist_ce_weight', default=0.4, type=float,
                        help='CE weight for the visual specialist logits during branch-specialist tuning')
    parser.add_argument('--bs_kd_weight', default=0.1, type=float,
                        help='Fused-teacher KD weight for branch-specialist tuning')
    parser.add_argument('--bs_temperature', default=2.0, type=float,
                        help='Temperature used by branch-specialist distillation')
    parser.add_argument('--bs_eval_audio_blend', default=0.0, type=float,
                        help='Blend audio specialist logits into validation audio logits during main training')
    parser.add_argument('--bs_eval_visual_blend', default=0.0, type=float,
                        help='Blend visual specialist logits into validation visual logits during main training')
    # ===== KS 相对 baseline（3）的关键新增 end: branch specialist 开关 =====
    parser.add_argument('--stage2_visual_tune', action='store_true',
                        help='Run stage-2 modality tuning from a pretrained checkpoint')
    parser.add_argument('--stage2_tune', dest='stage2_visual_tune', action='store_true',
                        help='Alias of --stage2_visual_tune')
    parser.add_argument('--stage2_init_ckpt', default='', type=str,
                        help='Checkpoint path used to initialize stage-2 modality tuning')
    parser.add_argument('--stage2_epochs', default=12, type=int,
                        help='Number of epochs for stage-2 modality tuning')
    parser.add_argument('--stage2_lr', default=1e-4, type=float,
                        help='Learning rate for stage-2 modality tuning')
    parser.add_argument('--stage2_lr_decay_step', default=6, type=int,
                        help='LR decay step for stage-2 modality tuning')
    parser.add_argument('--stage2_lr_decay_ratio', default=0.5, type=float,
                        help='LR decay ratio for stage-2 modality tuning')
    parser.add_argument('--stage2_target', default='visual', type=str,
                        choices=['visual', 'audio', 'both'],
                        help='Target modality branch for stage-2 tuning')
    parser.add_argument('--stage2_unfreeze_scope', default='layer4', type=str,
                        choices=['layer4', 'layer3_layer4', 'all_visual', 'all_target'],
                        help='Encoder scope to unfreeze in stage-2 tuning')
    parser.add_argument('--stage2_visual_ce_weight', default=1.0, type=float,
                        help='Weight for shared visual-logit CE during stage-2 tuning')
    parser.add_argument('--stage2_audio_ce_weight', default=1.0, type=float,
                        help='Weight for shared audio-logit CE during stage-2 tuning')
    parser.add_argument('--stage2_specialist_ce_weight', default=0.5, type=float,
                        help='Weight for specialist visual-logit CE during stage-2 tuning')
    parser.add_argument('--stage2_audio_specialist_ce_weight', default=0.5, type=float,
                        help='Weight for specialist audio-logit CE during stage-2 tuning')
    parser.add_argument('--stage2_kd_weight', default=0.1, type=float,
                        help='Weight for fused-teacher KD during stage-2 tuning')
    parser.add_argument('--stage2_fused_consistency_weight', default=0.1, type=float,
                        help='Weight for preserving fused predictions during stage-2 tuning')
    parser.add_argument('--stage2_temperature', default=2.0, type=float,
                        help='Temperature used by KD losses in stage-2 tuning')
    parser.add_argument('--stage2_eval_visual_blend', default=0.0, type=float,
                        help='Blend specialist visual logits into validation visual logits in stage-2')
    parser.add_argument('--stage2_eval_audio_blend', default=0.0, type=float,
                        help='Blend specialist audio logits into validation audio logits in stage-2')
    parser.add_argument('--stage2_save_metric', default='visual', type=str,
                        choices=['visual', 'audio', 'best', 'sum_unimodal'],
                        help='Metric used to select the best stage-2 checkpoint')
    parser.add_argument('--stage2_tune_shared_head', action='store_true',
                        help='Allow stage-2 to tune the shared fusion-head slice used by unimodal evaluation')
    parser.add_argument('--stage2_train_bias', action='store_true',
                        help='Also tune the shared fusion-head bias during stage-2 shared-head tuning')
    parser.add_argument('--stage2_acc_drop_tolerance', default=-1.0, type=float,
                        help='Maximum allowed fused-accuracy drop from the init checkpoint when selecting the best stage-2 model; negative disables the constraint')
    parser.add_argument('--stage2_audio_teacher_ckpt', default='', type=str,
                        help='Optional checkpoint path of an audio-only teacher used during stage-2 distillation')
    parser.add_argument('--stage2_visual_teacher_ckpt', default='', type=str,
                        help='Optional checkpoint path of a visual-only teacher used during stage-2 distillation')
    parser.add_argument('--stage2_export_path', default='', type=str,
                        help='Optional stable checkpoint path that is overwritten whenever stage-2 finds a better model')
    parser.add_argument('--stage2_adaptive_repair', action='store_true',
                        help='Use teacher-student confidence gaps to dynamically upweight the weaker modality during stage-2 joint repair')
    parser.add_argument('--stage2_repair_margin', default=0.01, type=float,
                        help='Dead-zone margin before adaptive stage-2 repair starts upweighting a modality')
    parser.add_argument('--stage2_repair_gain', default=6.0, type=float,
                        help='Gain used to convert confidence gaps into adaptive stage-2 repair weights')
    parser.add_argument('--stage2_repair_cap', default=2.5, type=float,
                        help='Maximum per-modality loss multiplier used by adaptive stage-2 repair')
    parser.add_argument('--stage2_consistency_floor', default=0.35, type=float,
                        help='Minimum fraction of fused-consistency weight kept during adaptive stage-2 repair')
    parser.add_argument('--input_mask_mode', default='none', type=str,
                        choices=['none', 'audio_only', 'visual_only'],
                        help='Optionally zero out one modality input during this run; useful for training unimodal teachers')
    parser.add_argument('--cremad_eval_sampling', default='uniform', type=str,
                        choices=['uniform', 'random'],
                        help='Frame sampling used by CREMAD test evaluation; uniform gives deterministic validation')
    parser.add_argument('--cremad_eval_num_views', default=1, type=int,
                        help='Number of deterministic visual views averaged during CREMAD evaluation')
    parser.add_argument('--run_cremad_pipeline', action='store_true',
                        help='Run the full CREMAD base->teacher->merge->eval pipeline from a single main.py command')
    parser.add_argument('--pipeline_base_epochs', default=50, type=int,
                        help='Base-training epochs used by --run_cremad_pipeline')
    parser.add_argument('--pipeline_eval_num_views', default=5, type=int,
                        help='Final deterministic-view count used by --run_cremad_pipeline')
    parser.add_argument('--pipeline_run_name', default='', type=str,
                        help='Optional run name used by pipeline logs')
    parser.add_argument('--run_ks_pipeline', action='store_true',
                        help='Run the full KineticSound audio-teacher -> visual-teacher -> joint-refine pipeline from a single main.py command')
    parser.add_argument('--ks_pipeline_base_ckpt',
                        default='ckpt/best_model_of_dataset_KineticSound_Ours_alpha_0.8_optimizer_sgd_training_epochs_0_epoch_49_acc_0.7300970873786408.pth',
                        type=str,
                        help='Base fused checkpoint used by --run_ks_pipeline')
    parser.add_argument('--ks_pipeline_audio_epochs', default=18, type=int,
                        help='Audio-teacher epochs used by --run_ks_pipeline')
    parser.add_argument('--ks_pipeline_visual_epochs', default=18, type=int,
                        help='Visual-teacher epochs used by --run_ks_pipeline')
    parser.add_argument('--ks_pipeline_joint_epochs', default=18, type=int,
                        help='Joint repair epochs used by --run_ks_pipeline')
    parser.add_argument('--ks_pipeline_profile', default='v9_fixed_frames', type=str,
                        choices=['v9_fixed_frames', 'v10_visual_push'],
                        help='Preset recipe used by --run_ks_pipeline')
    parser.add_argument('--use_single_run_repair', action='store_true',
                        help='Apply weaker-modality masked repair inside one from-scratch run')
    parser.add_argument('--sr_start_epoch', default=8, type=int,
                        help='Start epoch for single-run masked repair')
    parser.add_argument('--sr_end_epoch', default=49, type=int,
                        help='End epoch for single-run masked repair')
    parser.add_argument('--sr_target', default='auto', type=str,
                        choices=['auto', 'audio', 'visual', 'both'],
                        help='Target modality for masked repair')
    parser.add_argument('--sr_audio_scale', default=1.0, type=float,
                        help='Base strength for audio masked repair')
    parser.add_argument('--sr_visual_scale', default=0.6, type=float,
                        help='Base strength for visual masked repair')
    parser.add_argument('--sr_margin', default=0.015, type=float,
                        help='Confidence margin before auto masked repair activates')
    parser.add_argument('--sr_gain', default=6.0, type=float,
                        help='Gain used to convert confidence gaps into masked-repair weights')
    parser.add_argument('--sr_cap', default=2.5, type=float,
                        help='Maximum repair weight used by single-run masked repair')
    parser.add_argument('--sr_shared_ce_weight', default=0.15, type=float,
                        help='CE weight on shared unimodal logits during masked repair')
    parser.add_argument('--sr_specialist_ce_weight', default=0.15, type=float,
                        help='CE weight on specialist unimodal logits during masked repair')
    parser.add_argument('--sr_fused_ce_weight', default=0.0, type=float,
                        help='Optional CE weight on fused logits during masked repair')
    parser.add_argument('--sr_kd_weight', default=0.05, type=float,
                        help='KD weight from full fused teacher to masked logits')
    parser.add_argument('--sr_temperature', default=2.0, type=float,
                        help='Temperature used by single-run masked repair')
    parser.add_argument('--use_ks_single_run_recipe', action='store_true',
                        help='Use the recommended KineticSound single-run from-scratch recipe')
    parser.add_argument('--use_author_targets', action='store_true',
                        help='Track whether one epoch jointly exceeds the paper-reported Acc/Audio/Visual targets for the selected dataset')
    parser.add_argument('--target_acc', default=-1.0, type=float,
                        help='Target fused accuracy threshold; negative means unset')
    parser.add_argument('--target_audio_acc', default=-1.0, type=float,
                        help='Target audio accuracy threshold; negative means unset')
    parser.add_argument('--target_visual_acc', default=-1.0, type=float,
                        help='Target visual accuracy threshold; negative means unset')
    parser.add_argument('--save_goal_ckpt', action='store_true',
                        help='Save the best checkpoint that satisfies all three target metrics at the same epoch')
    parser.add_argument('--goal_ckpt_tag', default='goal', type=str,
                        help='Filename prefix used by --save_goal_ckpt')
    parser.add_argument('--goal_score_mode', default='min_margin', type=str,
                        choices=['min_margin', 'sum_margin'],
                        help='How to rank epochs for the three-metric goal; min_margin prioritizes the weakest metric')

    return parser.parse_args()


def apply_ks_single_run_recipe(args):
    if not getattr(args, 'use_ks_single_run_recipe', False):
        return args

    if args.dataset != 'KineticSound':
        print('[KSRecipe] skip preset because dataset={} is not KineticSound'.format(args.dataset))
        return args

    # ===== KS 相对 baseline（3）的关键新增 start: KS 单次运行的保守配方 =====
    # baseline（3）没有这一段。这里把当前验证有效的 KS 配方固化下来：
    # 1) 测试时默认至少 10 个 view；
    # 2) 开启 audio branch specialist；
    # 3) 用较小权重补 audio，而不是大幅重写训练目标。
    args.ks_eval_num_views = max(int(args.ks_eval_num_views), 10)
    args.use_branch_specialist = True
    args.bs_start_epoch = 8
    args.bs_target = 'audio'
    args.bs_audio_ce_weight = 0.05
    args.bs_visual_ce_weight = 0.0
    args.bs_audio_specialist_ce_weight = 0.20
    args.bs_visual_specialist_ce_weight = 0.0
    args.bs_kd_weight = 0.05
    args.bs_eval_audio_blend = 0.18
    # ===== KS 相对 baseline（3）的关键新增 end: KS 单次运行的保守配方 =====
    args.bs_eval_visual_blend = 0.0

    args.use_single_run_repair = True
    args.sr_start_epoch = 8
    args.sr_end_epoch = min(max(0, args.epochs - 1), 49)
    args.sr_target = 'auto'
    args.sr_audio_scale = 1.0
    args.sr_visual_scale = 0.55
    args.sr_margin = 0.015
    args.sr_gain = 6.0
    args.sr_cap = 2.5
    args.sr_shared_ce_weight = 0.12
    args.sr_specialist_ce_weight = 0.18
    args.sr_fused_ce_weight = 0.03
    args.sr_kd_weight = 0.05
    args.sr_temperature = 2.0
    return args


# 示例代码：冻结 audio_net 的所有参数
def freeze_audio_net(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.audio_net.parameters():
        param.requires_grad = False


def compute_trace_progress(trace_list, epoch, window=10):
    if epoch == 0:
        return 1.0
    if trace_list is None or len(trace_list) == 0:
        return 0.0

    usable_window = min(window, max(1, len(trace_list) // 2))
    if len(trace_list) < usable_window * 2:
        recent = float(np.mean(trace_list[-usable_window:]))
        previous = float(np.mean(trace_list[:-usable_window])) if len(trace_list) > usable_window else recent
    else:
        recent = float(np.mean(trace_list[-usable_window:]))
        previous = float(np.mean(trace_list[-2 * usable_window:-usable_window]))

    denominator = max(abs(recent), 1e-8)
    return (recent - previous) / denominator


def get_model_module(model):
    return model.module if isinstance(model, nn.DataParallel) else model



def apply_input_mask(spec, image, mode):
    if mode == 'audio_only':
        return spec, torch.zeros_like(image)
    if mode == 'visual_only':
        return torch.zeros_like(spec), image
    return spec, image



def forward_eval_logits(args, model, spec, image, epoch=None):
    # ===== KS 相对 baseline（3）的关键新增 =====
    # baseline（3）默认只支持单视角 image。
    # 现在如果 image 是 6 维张量，说明 KSDataset.py 已经构造好了多视角输入，
    # 这里就逐视角前向并平均 fused/audio/visual logits。
    if image.dim() != 6:
        a, v, a_output, v_output, out, _, _ = model(spec.unsqueeze(1).float(), image.float())
        out_a, out_v = get_unimodal_logits(args, model, a, v, a_output, v_output, apply_aux_blend=True, epoch=epoch)
        return out, out_a, out_v

    fused_logits = []
    audio_logits = []
    visual_logits = []
    for view_idx in range(image.shape[1]):
        a, v, a_output, v_output, out, _, _ = model(spec.unsqueeze(1).float(), image[:, view_idx].float())
        out_a, out_v = get_unimodal_logits(args, model, a, v, a_output, v_output, apply_aux_blend=True, epoch=epoch)
        fused_logits.append(out)
        audio_logits.append(out_a)
        visual_logits.append(out_v)

    out = torch.stack(fused_logits, dim=0).mean(dim=0)
    out_a = torch.stack(audio_logits, dim=0).mean(dim=0)
    out_v = torch.stack(visual_logits, dim=0).mean(dim=0)
    return out, out_a, out_v



def build_optional_teacher_model(args, ckpt_path, device, gpu_ids, label='teacher'):
    if not ckpt_path:
        return None

    print('[Stage2] loading {} checkpoint from {}'.format(label, ckpt_path))
    teacher = AVClassifier(args)
    teacher.apply(weight_init)
    teacher.to(device)
    teacher = torch.nn.DataParallel(teacher, device_ids=gpu_ids)
    teacher.cuda()
    load_checkpoint_into_model(teacher, ckpt_path, device, strict=False)
    freeze_module_parameters(get_model_module(teacher))
    teacher.eval()
    return teacher



def get_lrmo_state(lrmo_state):
    if lrmo_state is None:
        lrmo_state = {'u_audio': 0.5, 'u_visual': 0.5}
    lrmo_state.setdefault('u_audio', 0.5)
    lrmo_state.setdefault('u_visual', 0.5)
    return lrmo_state



def compute_lrmo_update_ratios(args, lrmo_state, conf_audio, conf_visual):
    lrmo_state = get_lrmo_state(lrmo_state)
    ema = float(np.clip(args.lrmo_ema, 0.0, 0.9999))

    lrmo_state['u_audio'] = ema * lrmo_state['u_audio'] + (1.0 - ema) * float(conf_audio)
    lrmo_state['u_visual'] = ema * lrmo_state['u_visual'] + (1.0 - ema) * float(conf_visual)

    significance = torch.tensor([
        lrmo_state['u_audio'],
        lrmo_state['u_visual'],
    ], dtype=torch.float32)
    tau = max(float(args.lrmo_tau), 1e-6)
    probs = torch.softmax(significance / tau, dim=0)
    ratios = 1.0 - probs

    ratio_floor = float(np.clip(args.lrmo_ratio_floor, 0.05, 0.99))
    ratio_ceiling = float(np.clip(args.lrmo_ratio_ceiling, ratio_floor, 1.0))
    ratios = torch.clamp(ratios, min=ratio_floor, max=ratio_ceiling)

    stats = {
        'u_audio': float(lrmo_state['u_audio']),
        'u_visual': float(lrmo_state['u_visual']),
        'ratio_audio': float(ratios[0].item()),
        'ratio_visual': float(ratios[1].item()),
    }
    return stats



def compute_lrmo_unit_importance(weight_grad):
    grad_square = weight_grad.detach().pow(2)
    if grad_square.dim() == 4:
        return grad_square.mean(dim=(1, 2, 3))
    if grad_square.dim() == 2:
        return grad_square.mean(dim=1)
    return grad_square.view(grad_square.size(0), -1).mean(dim=1)



def sample_lrmo_unit_mask(args, scores, keep_ratio):
    num_units = int(scores.numel())
    if num_units <= 1 or keep_ratio >= 0.999:
        return torch.ones_like(scores), num_units

    keep_count = max(1, int(np.ceil(float(keep_ratio) * num_units)))
    scores = scores.float()
    scores = scores - scores.min()
    probs = scores + 1e-8
    probs = probs / probs.sum()
    sampled_idx = torch.multinomial(probs, keep_count, replacement=False)

    scale = 1.0
    if getattr(args, 'lrmo_unbiased', False):
        scale = min(1.0 / max(float(keep_ratio), 1e-6), float(args.lrmo_scale_cap))

    mask = torch.zeros_like(scores)
    mask[sampled_idx] = scale
    return mask, keep_count



def apply_lrmo_gradient_mask(args, model, epoch, lrmo_stats, writer=None, importance_cache=None):
    zero_stats = {
        'ratio_audio': 1.0,
        'ratio_visual': 1.0,
        'u_audio': 0.0,
        'u_visual': 0.0,
        'applied_audio_modules': 0,
        'applied_visual_modules': 0,
        'audio_keep_avg': 1.0,
        'visual_keep_avg': 1.0,
    }
    if not getattr(args, 'use_lrmo', False):
        return zero_stats
    if epoch < args.lrmo_start_epoch or epoch > args.lrmo_end_epoch:
        zero_stats['u_audio'] = float(lrmo_stats['u_audio'])
        zero_stats['u_visual'] = float(lrmo_stats['u_visual'])
        return zero_stats

    module = get_model_module(model)
    ratio_audio = float(lrmo_stats['ratio_audio'])
    ratio_visual = float(lrmo_stats['ratio_visual'])
    audio_keep_values = []
    visual_keep_values = []
    applied_audio_modules = 0
    applied_visual_modules = 0

    for name, submodule in module.named_modules():
        if not isinstance(submodule, (nn.Conv2d, nn.Linear)):
            continue
        if submodule.weight.grad is None or not submodule.weight.requires_grad:
            continue

        is_audio_module = ('audio_net' in name) or (args.lrmo_include_heads and 'head_audio' in name)
        is_visual_module = ('visual_net' in name) or (args.lrmo_include_heads and 'head_video' in name)
        if not is_audio_module and not is_visual_module:
            continue

        keep_ratio = ratio_audio if is_audio_module else ratio_visual
        importance_tensor = submodule.weight.grad
        if args.lrmo_importance_source == 'fim' and importance_cache is not None and name in importance_cache:
            cached_tensor = importance_cache[name]
            if cached_tensor is not None and cached_tensor.numel() == submodule.weight.grad.numel():
                importance_tensor = cached_tensor
        importance_scores = compute_lrmo_unit_importance(importance_tensor)
        unit_mask, keep_count = sample_lrmo_unit_mask(args, importance_scores, keep_ratio)
        expanded_mask = unit_mask.view(unit_mask.size(0), *([1] * (submodule.weight.grad.dim() - 1)))
        submodule.weight.grad.mul_(expanded_mask)
        if submodule.bias is not None and submodule.bias.grad is not None:
            submodule.bias.grad.mul_(unit_mask)

        keep_fraction = float(keep_count) / float(max(1, importance_scores.numel()))
        if is_audio_module:
            applied_audio_modules += 1
            audio_keep_values.append(keep_fraction)
        else:
            applied_visual_modules += 1
            visual_keep_values.append(keep_fraction)

    mask_stats = {
        'ratio_audio': ratio_audio,
        'ratio_visual': ratio_visual,
        'u_audio': float(lrmo_stats['u_audio']),
        'u_visual': float(lrmo_stats['u_visual']),
        'applied_audio_modules': applied_audio_modules,
        'applied_visual_modules': applied_visual_modules,
        'audio_keep_avg': float(np.mean(audio_keep_values)) if audio_keep_values else 1.0,
        'visual_keep_avg': float(np.mean(visual_keep_values)) if visual_keep_values else 1.0,
    }

    if writer is not None:
        writer.add_scalar('lrmo/u_audio', mask_stats['u_audio'], epoch)
        writer.add_scalar('lrmo/u_visual', mask_stats['u_visual'], epoch)
        writer.add_scalar('lrmo/ratio_audio', mask_stats['ratio_audio'], epoch)
        writer.add_scalar('lrmo/ratio_visual', mask_stats['ratio_visual'], epoch)
        writer.add_scalar('lrmo/audio_keep_avg', mask_stats['audio_keep_avg'], epoch)
        writer.add_scalar('lrmo/visual_keep_avg', mask_stats['visual_keep_avg'], epoch)

    return mask_stats



def compute_branch_calibrated_logits(args, model, a, v, out_a_shared, out_v_shared, detach_anchor=False):
    module = get_model_module(model)
    anchor_a = out_a_shared.detach() if detach_anchor else out_a_shared
    anchor_v = out_v_shared.detach() if detach_anchor else out_v_shared

    audio_feature = a.detach()
    visual_feature = v.detach()
    aux_audio = module.head_audio(audio_feature)
    aux_visual = module.head_video(visual_feature)
    boost_audio = module.boost_head_audio(audio_feature)
    boost_visual = module.boost_head_video(visual_feature)

    calibrated_audio = anchor_a + args.bc_aux_scale * aux_audio + args.bc_residual_scale * boost_audio
    calibrated_visual = anchor_v + args.bc_aux_scale * aux_visual + args.bc_residual_scale * boost_visual
    return calibrated_audio, calibrated_visual, aux_audio, aux_visual, boost_audio, boost_visual


def get_unimodal_logits(args, model, a, v, audio_logits_aux=None, visual_logits_aux=None, apply_aux_blend=False, epoch=None):
    module = get_model_module(model)
    if args.fusion_method == 'sum':
        out_v = (torch.mm(v, torch.transpose(module.fusion_module.fc_y.weight, 0, 1)) +
                 module.fusion_module.fc_y.bias / 2)
        out_a = (torch.mm(a, torch.transpose(module.fusion_module.fc_x.weight, 0, 1)) +
                 module.fusion_module.fc_x.bias / 2)
    else:
        out_v = torch.mm(v, torch.transpose(module.head.weight[:, 512:], 0, 1)) + module.head.bias / 2
        out_a = torch.mm(a, torch.transpose(module.head.weight[:, :512], 0, 1)) + module.head.bias / 2

    if apply_aux_blend:
        rescue_blend = float(np.clip(getattr(args, 'rescue_eval_blend', 0.0), 0.0, 1.0))
        if getattr(args, 'use_dual_rescue', False) and rescue_blend > 0:
            if audio_logits_aux is not None:
                out_a = (1.0 - rescue_blend) * out_a + rescue_blend * audio_logits_aux
            if visual_logits_aux is not None:
                out_v = (1.0 - rescue_blend) * out_v + rescue_blend * visual_logits_aux

        bc_active_for_eval = branch_calibration_is_active(args, epoch) if epoch is not None else False
        if bc_active_for_eval:
            audio_blend = float(np.clip(getattr(args, 'bc_blend_audio', 0.0), 0.0, 1.0))
            visual_blend = float(np.clip(getattr(args, 'bc_blend_visual', 0.0), 0.0, 1.0))
            calibrated_audio, calibrated_visual, _, _, _, _ = compute_branch_calibrated_logits(
                args, model, a, v, out_a, out_v, detach_anchor=False
            )
            if audio_blend > 0:
                out_a = (1.0 - audio_blend) * out_a + audio_blend * calibrated_audio
            if visual_blend > 0:
                out_v = (1.0 - visual_blend) * out_v + visual_blend * calibrated_visual

        if branch_specialist_is_active(args, epoch):
            audio_blend = float(np.clip(getattr(args, 'bs_eval_audio_blend', 0.0), 0.0, 1.0))
            visual_blend = float(np.clip(getattr(args, 'bs_eval_visual_blend', 0.0), 0.0, 1.0))
            specialist_audio, specialist_visual, _, _ = get_specialist_logits(
                model, a, v, audio_logits_aux, visual_logits_aux
            )
            if audio_blend > 0:
                out_a = (1.0 - audio_blend) * out_a + audio_blend * specialist_audio
            if visual_blend > 0:
                out_v = (1.0 - visual_blend) * out_v + visual_blend * specialist_visual

        stage2_audio_blend = float(np.clip(getattr(args, 'stage2_eval_audio_blend', 0.0), 0.0, 1.0))
        if stage2_audio_blend > 0:
            specialist_audio, _, boost_audio, _ = get_specialist_logits(model, a, v, audio_logits_aux, visual_logits_aux)
            out_a = (1.0 - stage2_audio_blend) * out_a + stage2_audio_blend * specialist_audio

        stage2_visual_blend = float(np.clip(getattr(args, 'stage2_eval_visual_blend', 0.0), 0.0, 1.0))
        if stage2_visual_blend > 0:
            _, specialist_visual, _, boost_visual = get_specialist_logits(model, a, v, audio_logits_aux, visual_logits_aux)
            out_v = (1.0 - stage2_visual_blend) * out_v + stage2_visual_blend * specialist_visual

    return out_a, out_v


def branch_calibration_is_active(args, epoch):
    return getattr(args, 'use_branch_calibration', False) and args.bc_start_epoch <= epoch <= args.bc_end_epoch


def compute_branch_calibration_loss(args, epoch, model, a, v, out, out_a_shared, out_v_shared, label):
    zero = out.new_tensor(0.0)
    if not branch_calibration_is_active(args, epoch):
        return zero, None

    calib_audio, calib_visual, aux_audio, aux_visual, boost_audio, boost_visual = compute_branch_calibrated_logits(
        args=args,
        model=model,
        a=a,
        v=v,
        out_a_shared=out_a_shared,
        out_v_shared=out_v_shared,
        detach_anchor=True,
    )

    ce_audio = F.cross_entropy(calib_audio, label)
    ce_visual = F.cross_entropy(calib_visual, label)
    temperature = max(args.bc_temperature, 1e-6)

    with torch.no_grad():
        teacher_prob = F.softmax(out.detach() / temperature, dim=1)
        teacher_conf = F.softmax(out.detach(), dim=1).gather(1, label.view(-1, 1)).mean().item()
        shared_conf_audio = F.softmax(out_a_shared.detach(), dim=1).gather(1, label.view(-1, 1)).mean().item()
        shared_conf_visual = F.softmax(out_v_shared.detach(), dim=1).gather(1, label.view(-1, 1)).mean().item()
        boosted_conf_audio = F.softmax(calib_audio.detach(), dim=1).gather(1, label.view(-1, 1)).mean().item()
        boosted_conf_visual = F.softmax(calib_visual.detach(), dim=1).gather(1, label.view(-1, 1)).mean().item()

        if args.bc_target == 'audio':
            weight_audio = args.bc_audio_scale
            weight_visual = 0.0
        elif args.bc_target == 'visual':
            weight_audio = 0.0
            weight_visual = args.bc_visual_scale
        elif args.bc_target == 'both':
            weight_audio = args.bc_audio_scale
            weight_visual = args.bc_visual_scale
        else:
            audio_is_weak = (shared_conf_visual - args.bc_sigma * shared_conf_audio) > args.bc_tau
            visual_is_weak = (shared_conf_audio - args.bc_sigma * shared_conf_visual) > args.bc_tau

            weight_audio = 0.0
            weight_visual = 0.0
            if audio_is_weak:
                weight_audio = args.bc_audio_scale * (1.0 + max(0.0, teacher_conf - shared_conf_audio - args.bc_auto_margin))
            if visual_is_weak:
                weight_visual = args.bc_visual_scale * (1.0 + max(0.0, teacher_conf - shared_conf_visual - args.bc_auto_margin))

    kd_audio = F.kl_div(
        F.log_softmax(calib_audio / temperature, dim=1),
        teacher_prob,
        reduction='batchmean'
    ) * (temperature ** 2)
    kd_visual = F.kl_div(
        F.log_softmax(calib_visual / temperature, dim=1),
        teacher_prob,
        reduction='batchmean'
    ) * (temperature ** 2)

    if weight_audio == 0.0 and weight_visual == 0.0:
        stats = {
            'bc_weight_audio': 0.0,
            'bc_weight_visual': 0.0,
            'bc_teacher_conf': float(teacher_conf),
            'bc_shared_conf_audio': float(shared_conf_audio),
            'bc_shared_conf_visual': float(shared_conf_visual),
            'bc_boosted_conf_audio': float(boosted_conf_audio),
            'bc_boosted_conf_visual': float(boosted_conf_visual),
            'bc_loss': 0.0,
        }
        return zero, stats

    loss = args.bc_ce_weight * (weight_audio * ce_audio + weight_visual * ce_visual)
    loss = loss + args.bc_kd_weight * (weight_audio * kd_audio + weight_visual * kd_visual)

    stats = {
        'bc_weight_audio': float(weight_audio),
        'bc_weight_visual': float(weight_visual),
        'bc_teacher_conf': float(teacher_conf),
        'bc_shared_conf_audio': float(shared_conf_audio),
        'bc_shared_conf_visual': float(shared_conf_visual),
        'bc_boosted_conf_audio': float(boosted_conf_audio),
        'bc_boosted_conf_visual': float(boosted_conf_visual),
        'bc_loss': float(loss.detach().item()),
    }
    return loss, stats


def load_checkpoint_into_model(model, ckpt_path, device, strict=False):
    if not ckpt_path:
        raise ValueError('Checkpoint path is required for loading.')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError('Checkpoint not found: {}'.format(ckpt_path))

    loaded_dict = torch.load(ckpt_path, map_location=device)
    state_dict = loaded_dict['model'] if isinstance(loaded_dict, dict) and 'model' in loaded_dict else loaded_dict
    load_result = model.load_state_dict(state_dict, strict=strict)

    missing_keys = list(getattr(load_result, 'missing_keys', []))
    unexpected_keys = list(getattr(load_result, 'unexpected_keys', []))
    if missing_keys:
        print('Checkpoint missing keys: {}'.format(len(missing_keys)))
    if unexpected_keys:
        print('Checkpoint unexpected keys: {}'.format(len(unexpected_keys)))

    return loaded_dict


def freeze_module_parameters(module):
    for param in module.parameters():
        param.requires_grad = False



def unfreeze_module_parameters(module):
    for param in module.parameters():
        param.requires_grad = True



def get_stage2_modality_train_modules(module, scope, target):
    train_modules = []

    if target in ['audio', 'both']:
        if scope == 'layer4':
            train_modules.append(module.audio_net.layer4)
        elif scope == 'layer3_layer4':
            train_modules.extend([module.audio_net.layer3, module.audio_net.layer4])
        elif scope in ['all_visual', 'all_target']:
            train_modules.append(module.audio_net)
        else:
            raise ValueError('Unsupported stage2_unfreeze_scope: {}'.format(scope))

    if target in ['visual', 'both']:
        if scope == 'layer4':
            train_modules.append(module.visual_net.layer4)
        elif scope == 'layer3_layer4':
            train_modules.extend([module.visual_net.layer3, module.visual_net.layer4])
        elif scope in ['all_visual', 'all_target']:
            train_modules.append(module.visual_net)
        else:
            raise ValueError('Unsupported stage2_unfreeze_scope: {}'.format(scope))

    return train_modules



def prepare_stage2_shared_head(module, args):
    if not getattr(args, 'stage2_tune_shared_head', False):
        return

    if args.fusion_method == 'concat':
        module.head.weight.requires_grad = True
        if getattr(module.head, 'bias', None) is not None and getattr(args, 'stage2_train_bias', False):
            module.head.bias.requires_grad = True
        return

    if args.fusion_method == 'sum':
        if args.stage2_target in ['audio', 'both']:
            unfreeze_module_parameters(module.fusion_module.fc_x)
        if args.stage2_target in ['visual', 'both']:
            unfreeze_module_parameters(module.fusion_module.fc_y)
        return

    unfreeze_module_parameters(module.fusion_module)



def apply_stage2_shared_head_gradient_mask(args, model):
    if not getattr(args, 'stage2_tune_shared_head', False):
        return

    module = get_model_module(model)
    if args.fusion_method == 'concat':
        weight_grad = getattr(module.head.weight, 'grad', None)
        if weight_grad is not None and args.stage2_target in ['audio', 'visual']:
            half = weight_grad.size(1) // 2
            if args.stage2_target == 'audio':
                weight_grad[:, half:] = 0.0
            else:
                weight_grad[:, :half] = 0.0

        if not getattr(args, 'stage2_train_bias', False) and getattr(module.head, 'bias', None) is not None:
            if module.head.bias.grad is not None:
                module.head.bias.grad.zero_()
        return

    if args.fusion_method == 'sum':
        if args.stage2_target == 'audio':
            if module.fusion_module.fc_y.weight.grad is not None:
                module.fusion_module.fc_y.weight.grad.zero_()
            if module.fusion_module.fc_y.bias is not None and module.fusion_module.fc_y.bias.grad is not None:
                module.fusion_module.fc_y.bias.grad.zero_()
        elif args.stage2_target == 'visual':
            if module.fusion_module.fc_x.weight.grad is not None:
                module.fusion_module.fc_x.weight.grad.zero_()
            if module.fusion_module.fc_x.bias is not None and module.fusion_module.fc_x.bias.grad is not None:
                module.fusion_module.fc_x.bias.grad.zero_()



def prepare_stage2_student(model, args):
    module = get_model_module(model)
    freeze_module_parameters(module)

    train_modules = get_stage2_modality_train_modules(module, args.stage2_unfreeze_scope, args.stage2_target)
    for train_module in train_modules:
        unfreeze_module_parameters(train_module)

    if args.stage2_target in ['audio', 'both']:
        unfreeze_module_parameters(module.head_audio)
        unfreeze_module_parameters(module.boost_head_audio)
    if args.stage2_target in ['visual', 'both']:
        unfreeze_module_parameters(module.head_video)
        unfreeze_module_parameters(module.boost_head_video)

    prepare_stage2_shared_head(module, args)

    trainable_names = [name for name, param in module.named_parameters() if param.requires_grad]
    print('Stage-2 trainable parameter groups: {}'.format(len(trainable_names)))
    for name in trainable_names:
        print('  [stage2] {}'.format(name))

    return trainable_names



def set_stage2_train_mode(model, args):
    module = get_model_module(model)
    module.train()
    module.audio_net.eval()
    module.visual_net.eval()
    module.head.eval()
    module.head2.eval()
    module.head_audio.eval()
    module.head_video.eval()
    module.boost_head_audio.eval()
    module.boost_head_video.eval()
    if hasattr(module, 'fusion_module'):
        module.fusion_module.eval()

    if getattr(args, 'stage2_tune_shared_head', False):
        if args.fusion_method == 'concat':
            module.head.train()
        elif hasattr(module, 'fusion_module'):
            module.fusion_module.train()

    for train_module in get_stage2_modality_train_modules(module, args.stage2_unfreeze_scope, args.stage2_target):
        train_module.train()

    if args.stage2_target in ['audio', 'both']:
        module.head_audio.train()
        module.boost_head_audio.train()
    if args.stage2_target in ['visual', 'both']:
        module.head_video.train()
        module.boost_head_video.train()



def build_stage2_optimizer(args, model):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError('No trainable parameters found for stage-2 modality tuning.')

    if args.optimizer == 'adam':
        return optim.Adam(trainable_params, lr=args.stage2_lr, weight_decay=1e-4)
    return optim.SGD(trainable_params, lr=args.stage2_lr, momentum=0.9, weight_decay=1e-4)



def stage2_visual_epoch(args, epoch, model, teacher_model, device, dataloader, optimizer,
                        scheduler=None, writer=None, audio_teacher_model=None, visual_teacher_model=None):
    set_stage2_train_mode(model, args)
    teacher_model.eval()
    module = get_model_module(model)
    temperature = max(args.stage2_temperature, 1e-6)

    total_loss = 0.0
    total_audio_ce = 0.0
    total_visual_ce = 0.0
    total_audio_specialist_ce = 0.0
    total_visual_specialist_ce = 0.0
    total_kd = 0.0
    total_consistency = 0.0
    total_audio_weight = 0.0
    total_visual_weight = 0.0
    total_consistency_scale = 0.0

    for step, (spec, image, label) in enumerate(dataloader):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        spec, image = apply_input_mask(spec, image, args.input_mask_mode)

        optimizer.zero_grad()

        a, v, out_audio_aux, out_visual_aux, out, _, _ = model(spec.unsqueeze(1).float(), image.float())
        out_a_shared, out_v_shared = get_unimodal_logits(args, model, a, v, out_audio_aux, out_visual_aux, apply_aux_blend=False)
        specialist_audio, specialist_visual, boost_audio, boost_visual = get_specialist_logits(
            model, a, v, out_audio_aux, out_visual_aux
        )

        with torch.no_grad():
            teacher_a, teacher_v, teacher_audio_aux, teacher_visual_aux, teacher_out, _, _ = teacher_model(spec.unsqueeze(1).float(), image.float())
            teacher_audio_shared, teacher_visual_shared = get_unimodal_logits(
                args, teacher_model, teacher_a, teacher_v, teacher_audio_aux, teacher_visual_aux, apply_aux_blend=False
            )
            teacher_prob = F.softmax(teacher_out / temperature, dim=1)
            teacher_audio_prob = F.softmax(teacher_audio_shared / temperature, dim=1)
            teacher_visual_prob = F.softmax(teacher_visual_shared / temperature, dim=1)

            if audio_teacher_model is not None:
                teacher_spec_audio, teacher_image_audio = apply_input_mask(spec, image, 'audio_only')
                audio_a, audio_v, audio_aux_a, audio_aux_v, _, _, _ = audio_teacher_model(
                    teacher_spec_audio.unsqueeze(1).float(), teacher_image_audio.float()
                )
                teacher_audio_shared, _ = get_unimodal_logits(
                    args, audio_teacher_model, audio_a, audio_v, audio_aux_a, audio_aux_v, apply_aux_blend=False
                )
                teacher_audio_prob = F.softmax(teacher_audio_shared / temperature, dim=1)

            if visual_teacher_model is not None:
                teacher_spec_visual, teacher_image_visual = apply_input_mask(spec, image, 'visual_only')
                visual_a, visual_v, visual_aux_a, visual_aux_v, _, _, _ = visual_teacher_model(
                    teacher_spec_visual.unsqueeze(1).float(), teacher_image_visual.float()
                )
                _, teacher_visual_shared = get_unimodal_logits(
                    args, visual_teacher_model, visual_a, visual_v, visual_aux_a, visual_aux_v, apply_aux_blend=False
                )
                teacher_visual_prob = F.softmax(teacher_visual_shared / temperature, dim=1)

        loss = out.new_tensor(0.0)
        kd_terms = []
        audio_loss_weight = 1.0
        visual_loss_weight = 1.0
        consistency_scale = 1.0

        if getattr(args, 'stage2_adaptive_repair', False) and args.stage2_target == 'both':
            with torch.no_grad():
                label_index = label.view(-1, 1)
                student_audio_conf = F.softmax(out_a_shared.detach(), dim=1).gather(1, label_index).mean().item()
                student_visual_conf = F.softmax(out_v_shared.detach(), dim=1).gather(1, label_index).mean().item()
                teacher_audio_conf = teacher_audio_prob.gather(1, label_index).mean().item()
                teacher_visual_conf = teacher_visual_prob.gather(1, label_index).mean().item()

                repair_margin = max(0.0, float(args.stage2_repair_margin))
                repair_gain = max(0.0, float(args.stage2_repair_gain))
                repair_cap = max(1.0, float(args.stage2_repair_cap))

                audio_teacher_gap = max(0.0, teacher_audio_conf - student_audio_conf - repair_margin)
                visual_teacher_gap = max(0.0, teacher_visual_conf - student_visual_conf - repair_margin)
                audio_cross_gap = max(0.0, student_visual_conf - student_audio_conf - repair_margin)
                visual_cross_gap = max(0.0, student_audio_conf - student_visual_conf - repair_margin)

                audio_priority = audio_teacher_gap + 0.5 * audio_cross_gap
                visual_priority = visual_teacher_gap + 0.5 * visual_cross_gap

                audio_loss_weight = min(repair_cap, 1.0 + repair_gain * audio_priority)
                visual_loss_weight = min(repair_cap, 1.0 + repair_gain * visual_priority)

                consistency_scale = 1.0 / max(audio_loss_weight, visual_loss_weight, 1.0)
                consistency_scale = max(float(args.stage2_consistency_floor), float(consistency_scale))

        audio_ce = out.new_tensor(0.0)
        audio_specialist_ce = out.new_tensor(0.0)
        if args.stage2_target in ['audio', 'both']:
            audio_ce = F.cross_entropy(out_a_shared, label)
            audio_specialist_ce = 0.5 * F.cross_entropy(out_audio_aux, label) + 0.5 * F.cross_entropy(boost_audio, label)
            kd_audio = F.kl_div(
                F.log_softmax(out_a_shared / temperature, dim=1),
                teacher_audio_prob,
                reduction='batchmean'
            ) * (temperature ** 2)
            kd_audio_specialist = F.kl_div(
                F.log_softmax(specialist_audio / temperature, dim=1),
                teacher_audio_prob,
                reduction='batchmean'
            ) * (temperature ** 2)
            loss = loss + audio_loss_weight * args.stage2_audio_ce_weight * audio_ce
            loss = loss + audio_loss_weight * args.stage2_audio_specialist_ce_weight * audio_specialist_ce
            kd_terms.extend([audio_loss_weight * kd_audio, audio_loss_weight * kd_audio_specialist])

        visual_ce = out.new_tensor(0.0)
        visual_specialist_ce = out.new_tensor(0.0)
        if args.stage2_target in ['visual', 'both']:
            visual_ce = F.cross_entropy(out_v_shared, label)
            visual_specialist_ce = 0.5 * F.cross_entropy(out_visual_aux, label) + 0.5 * F.cross_entropy(boost_visual, label)
            kd_visual = F.kl_div(
                F.log_softmax(out_v_shared / temperature, dim=1),
                teacher_visual_prob,
                reduction='batchmean'
            ) * (temperature ** 2)
            kd_visual_specialist = F.kl_div(
                F.log_softmax(specialist_visual / temperature, dim=1),
                teacher_visual_prob,
                reduction='batchmean'
            ) * (temperature ** 2)
            loss = loss + visual_loss_weight * args.stage2_visual_ce_weight * visual_ce
            loss = loss + visual_loss_weight * args.stage2_specialist_ce_weight * visual_specialist_ce
            kd_terms.extend([visual_loss_weight * kd_visual, visual_loss_weight * kd_visual_specialist])

        kd_term = out.new_tensor(0.0)
        if len(kd_terms) > 0:
            kd_term = sum(kd_terms) / len(kd_terms)
            loss = loss + args.stage2_kd_weight * kd_term

        fused_consistency = F.kl_div(
            F.log_softmax(out / temperature, dim=1),
            teacher_prob,
            reduction='batchmean'
        ) * (temperature ** 2)
        loss = loss + (args.stage2_fused_consistency_weight * consistency_scale) * fused_consistency

        loss.backward()
        apply_stage2_shared_head_gradient_mask(args, model)
        optimizer.step()

        total_loss += loss.item()
        total_audio_ce += audio_ce.item()
        total_visual_ce += visual_ce.item()
        total_audio_specialist_ce += audio_specialist_ce.item()
        total_visual_specialist_ce += visual_specialist_ce.item()
        total_kd += kd_term.item()
        total_consistency += fused_consistency.item()
        total_audio_weight += float(audio_loss_weight)
        total_visual_weight += float(visual_loss_weight)
        total_consistency_scale += float(consistency_scale)

    if scheduler is not None:
        scheduler.step()

    num_steps = max(len(dataloader), 1)
    stats = {
        'loss': total_loss / num_steps,
        'audio_ce': total_audio_ce / num_steps,
        'visual_ce': total_visual_ce / num_steps,
        'audio_specialist_ce': total_audio_specialist_ce / num_steps,
        'visual_specialist_ce': total_visual_specialist_ce / num_steps,
        'kd': total_kd / num_steps,
        'consistency': total_consistency / num_steps,
        'audio_weight': total_audio_weight / num_steps,
        'visual_weight': total_visual_weight / num_steps,
        'consistency_scale': total_consistency_scale / num_steps,
    }

    if writer is not None:
        writer.add_scalar('stage2/loss', stats['loss'], epoch)
        writer.add_scalar('stage2/audio_ce', stats['audio_ce'], epoch)
        writer.add_scalar('stage2/visual_ce', stats['visual_ce'], epoch)
        writer.add_scalar('stage2/audio_specialist_ce', stats['audio_specialist_ce'], epoch)
        writer.add_scalar('stage2/visual_specialist_ce', stats['visual_specialist_ce'], epoch)
        writer.add_scalar('stage2/kd', stats['kd'], epoch)
        writer.add_scalar('stage2/consistency', stats['consistency'], epoch)
        writer.add_scalar('stage2/audio_weight', stats['audio_weight'], epoch)
        writer.add_scalar('stage2/visual_weight', stats['visual_weight'], epoch)
        writer.add_scalar('stage2/consistency_scale', stats['consistency_scale'], epoch)

    return stats



def save_stage2_checkpoint(args, model, optimizer, scheduler, epoch, acc, acc_a, acc_v, metric_value):
    os.makedirs(args.ckpt_path, exist_ok=True)
    model_name = 'stage2_{}_{}_scope_{}_epoch_{}_acc_{:.4f}_audio_{:.4f}_visual_{:.4f}.pth'.format(
        args.stage2_target,
        args.dataset,
        args.stage2_unfreeze_scope,
        epoch,
        acc,
        acc_a,
        acc_v,
    )
    save_dir = os.path.join(args.ckpt_path, model_name)
    saved_dict = {
        'saved_epoch': epoch,
        'modulation': args.modulation,
        'alpha': args.alpha,
        'fusion': args.fusion_method,
        'acc': acc,
        'acc_a': acc_a,
        'acc_v': acc_v,
        'stage2_target': args.stage2_target,
        'stage2_metric_name': args.stage2_save_metric,
        'stage2_metric_value': metric_value,
        'stage2_audio_blend': args.stage2_eval_audio_blend,
        'stage2_visual_blend': args.stage2_eval_visual_blend,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(saved_dict, save_dir)
    if getattr(args, 'stage2_export_path', ''):
        export_dir = os.path.dirname(args.stage2_export_path)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
        torch.save(saved_dict, args.stage2_export_path)
    return save_dir



def compute_dual_rescue_loss(args, epoch, out, out_audio_aux, out_visual_aux, label, k_audio, k_visual, device):
    temperature = max(args.rescue_temperature, 1e-6)
    rescue_progress = min(1.0, epoch / max(1, args.lr_decay_step))

    aux_loss_audio = F.cross_entropy(out_audio_aux, label)
    aux_loss_visual = F.cross_entropy(out_visual_aux, label)

    with torch.no_grad():
        fused_prob = F.softmax(out, dim=1)
        audio_prob = F.softmax(out_audio_aux, dim=1)
        visual_prob = F.softmax(out_visual_aux, dim=1)

        conf_audio = audio_prob.gather(1, label.view(-1, 1)).mean().item()
        conf_visual = visual_prob.gather(1, label.view(-1, 1)).mean().item()

        audio_lag = max(0.0, (conf_visual - conf_audio) - args.rescue_margin)
        visual_lag = max(0.0, (conf_audio - conf_visual) - args.rescue_margin)

        audio_lag += args.rescue_progress_weight * max(0.0, k_visual - k_audio)
        visual_lag += args.rescue_progress_weight * max(0.0, k_audio - k_visual)

        rescue_weight_audio = 1.0 + audio_lag
        rescue_weight_visual = 1.0 + visual_lag
        teacher_prob = F.softmax(out.detach() / temperature, dim=1)
        teacher_conf = fused_prob.gather(1, label.view(-1, 1)).mean().item()

    distill_audio = F.kl_div(
        F.log_softmax(out_audio_aux / temperature, dim=1),
        teacher_prob,
        reduction='batchmean'
    ) * (temperature ** 2)
    distill_visual = F.kl_div(
        F.log_softmax(out_visual_aux / temperature, dim=1),
        teacher_prob,
        reduction='batchmean'
    ) * (temperature ** 2)

    rescue_ce = args.rescue_ce_weight * (
        rescue_weight_audio * aux_loss_audio + rescue_weight_visual * aux_loss_visual
    )
    rescue_kd = args.rescue_kd_weight * rescue_progress * teacher_conf * (
        rescue_weight_audio * distill_audio + rescue_weight_visual * distill_visual
    )
    rescue_loss = rescue_ce + rescue_kd

    rescue_stats = {
        'conf_audio': conf_audio,
        'conf_visual': conf_visual,
        'rescue_weight_audio': rescue_weight_audio,
        'rescue_weight_visual': rescue_weight_visual,
        'teacher_conf': teacher_conf,
        'rescue_progress': rescue_progress,
        'rescue_loss': rescue_loss.detach().item(),
        'rescue_ce': rescue_ce.detach().item(),
        'rescue_kd': rescue_kd.detach().item(),
    }

    return rescue_loss, rescue_stats



def compute_branch_specialist_loss(args, epoch, model, a, v, out, out_a_shared, out_v_shared,
                                   out_audio_aux, out_visual_aux, label):
    zero = out.new_tensor(0.0)
    if not branch_specialist_is_active(args, epoch):
        return zero, None

    # ===== KS 相对 baseline（3）的关键新增 =====
    # baseline（3）没有这一支。这里的思路不是推翻原有融合训练，
    # 而是额外给单模态一个“保守的专家分支损失”：
    # - shared CE: 保持原单模态头可用
    # - specialist CE: 让补充分支学到更灵活的单模态判别
    # - KD: 用融合输出当 teacher，避免专家头完全跑偏
    temperature = max(args.bs_temperature, 1e-6)
    specialist_audio, specialist_visual, boost_audio, boost_visual = get_specialist_logits(
        model, a, v, out_audio_aux, out_visual_aux
    )

    with torch.no_grad():
        teacher_prob = F.softmax(out.detach() / temperature, dim=1)
        teacher_conf = F.softmax(out.detach(), dim=1).gather(1, label.view(-1, 1)).mean().item()

    loss = zero
    kd_terms = []
    audio_shared_ce = zero
    visual_shared_ce = zero
    audio_specialist_ce = zero
    visual_specialist_ce = zero

    if args.bs_target in ['audio', 'both']:
        if args.bs_audio_ce_weight > 0:
            audio_shared_ce = F.cross_entropy(out_a_shared, label)
            loss = loss + args.bs_audio_ce_weight * audio_shared_ce

        if args.bs_audio_specialist_ce_weight > 0:
            audio_specialist_ce = 0.5 * F.cross_entropy(out_audio_aux, label) + 0.5 * F.cross_entropy(boost_audio, label)
            loss = loss + args.bs_audio_specialist_ce_weight * audio_specialist_ce

        kd_audio = F.kl_div(
            F.log_softmax(specialist_audio / temperature, dim=1),
            teacher_prob,
            reduction='batchmean'
        ) * (temperature ** 2)
        kd_terms.append(kd_audio)

    if args.bs_target in ['visual', 'both']:
        if args.bs_visual_ce_weight > 0:
            visual_shared_ce = F.cross_entropy(out_v_shared, label)
            loss = loss + args.bs_visual_ce_weight * visual_shared_ce

        if args.bs_visual_specialist_ce_weight > 0:
            visual_specialist_ce = 0.5 * F.cross_entropy(out_visual_aux, label) + 0.5 * F.cross_entropy(boost_visual, label)
            loss = loss + args.bs_visual_specialist_ce_weight * visual_specialist_ce

        kd_visual = F.kl_div(
            F.log_softmax(specialist_visual / temperature, dim=1),
            teacher_prob,
            reduction='batchmean'
        ) * (temperature ** 2)
        kd_terms.append(kd_visual)

    kd_term = zero
    if args.bs_kd_weight > 0 and len(kd_terms) > 0:
        kd_term = sum(kd_terms) / len(kd_terms)
        loss = loss + args.bs_kd_weight * kd_term

    stats = {
        'bs_loss': loss.detach().item(),
        'bs_audio_shared_ce': audio_shared_ce.detach().item(),
        'bs_visual_shared_ce': visual_shared_ce.detach().item(),
        'bs_audio_specialist_ce': audio_specialist_ce.detach().item(),
        'bs_visual_specialist_ce': visual_specialist_ce.detach().item(),
        'bs_kd': kd_term.detach().item(),
        'bs_teacher_conf': teacher_conf,
    }
    return loss, stats


def compute_single_run_repair_loss(args, epoch, model, spec, image, out, label, conf_audio_batch, conf_visual_batch):
    zero = out.new_tensor(0.0)
    if not single_run_repair_is_active(args, epoch):
        return zero, None

    repair_weights = resolve_single_run_repair_weights(args, conf_audio_batch, conf_visual_batch)
    weight_audio = float(repair_weights['weight_audio'])
    weight_visual = float(repair_weights['weight_visual'])
    if weight_audio <= 0 and weight_visual <= 0:
        return zero, {
            'sr_loss': 0.0,
            'sr_weight_audio': 0.0,
            'sr_weight_visual': 0.0,
            'sr_gap_audio': float(repair_weights['gap_audio']),
            'sr_gap_visual': float(repair_weights['gap_visual']),
            'sr_audio_shared_ce': 0.0,
            'sr_visual_shared_ce': 0.0,
            'sr_audio_specialist_ce': 0.0,
            'sr_visual_specialist_ce': 0.0,
            'sr_audio_fused_ce': 0.0,
            'sr_visual_fused_ce': 0.0,
            'sr_audio_kd': 0.0,
            'sr_visual_kd': 0.0,
            'sr_teacher_conf': float(F.softmax(out.detach(), dim=1).gather(1, label.view(-1, 1)).mean().item()),
        }

    temperature = max(args.sr_temperature, 1e-6)
    with torch.no_grad():
        teacher_prob = F.softmax(out.detach() / temperature, dim=1)
        teacher_conf = F.softmax(out.detach(), dim=1).gather(1, label.view(-1, 1)).mean().item()

    loss = zero
    audio_shared_ce = zero
    visual_shared_ce = zero
    audio_specialist_ce = zero
    visual_specialist_ce = zero
    audio_fused_ce = zero
    visual_fused_ce = zero
    audio_kd = zero
    visual_kd = zero

    if weight_audio > 0:
        spec_audio, image_audio = apply_input_mask(spec, image, 'audio_only')
        a_audio, v_audio, out_audio_aux, out_visual_aux, out_audio_masked, _, _ = model(
            spec_audio.unsqueeze(1).float(), image_audio.float()
        )
        out_a_masked, _ = get_unimodal_logits(
            args, model, a_audio, v_audio, out_audio_aux, out_visual_aux, apply_aux_blend=False
        )
        specialist_audio, _, boost_audio, _ = get_specialist_logits(
            model, a_audio, v_audio, out_audio_aux, out_visual_aux
        )

        if args.sr_shared_ce_weight > 0:
            audio_shared_ce = F.cross_entropy(out_a_masked, label)
            loss = loss + weight_audio * args.sr_shared_ce_weight * audio_shared_ce
        if args.sr_specialist_ce_weight > 0:
            audio_specialist_ce = 0.5 * F.cross_entropy(out_audio_aux, label) + 0.5 * F.cross_entropy(boost_audio, label)
            loss = loss + weight_audio * args.sr_specialist_ce_weight * audio_specialist_ce
        if args.sr_fused_ce_weight > 0:
            audio_fused_ce = F.cross_entropy(out_audio_masked, label)
            loss = loss + weight_audio * args.sr_fused_ce_weight * audio_fused_ce
        if args.sr_kd_weight > 0:
            audio_kd_shared = F.kl_div(
                F.log_softmax(out_a_masked / temperature, dim=1),
                teacher_prob,
                reduction='batchmean'
            ) * (temperature ** 2)
            audio_kd_specialist = F.kl_div(
                F.log_softmax(specialist_audio / temperature, dim=1),
                teacher_prob,
                reduction='batchmean'
            ) * (temperature ** 2)
            audio_kd = 0.5 * audio_kd_shared + 0.5 * audio_kd_specialist
            loss = loss + weight_audio * args.sr_kd_weight * audio_kd

    if weight_visual > 0:
        spec_visual, image_visual = apply_input_mask(spec, image, 'visual_only')
        a_visual, v_visual, out_audio_aux, out_visual_aux, out_visual_masked, _, _ = model(
            spec_visual.unsqueeze(1).float(), image_visual.float()
        )
        _, out_v_masked = get_unimodal_logits(
            args, model, a_visual, v_visual, out_audio_aux, out_visual_aux, apply_aux_blend=False
        )
        _, specialist_visual, _, boost_visual = get_specialist_logits(
            model, a_visual, v_visual, out_audio_aux, out_visual_aux
        )

        if args.sr_shared_ce_weight > 0:
            visual_shared_ce = F.cross_entropy(out_v_masked, label)
            loss = loss + weight_visual * args.sr_shared_ce_weight * visual_shared_ce
        if args.sr_specialist_ce_weight > 0:
            visual_specialist_ce = 0.5 * F.cross_entropy(out_visual_aux, label) + 0.5 * F.cross_entropy(boost_visual, label)
            loss = loss + weight_visual * args.sr_specialist_ce_weight * visual_specialist_ce
        if args.sr_fused_ce_weight > 0:
            visual_fused_ce = F.cross_entropy(out_visual_masked, label)
            loss = loss + weight_visual * args.sr_fused_ce_weight * visual_fused_ce
        if args.sr_kd_weight > 0:
            visual_kd_shared = F.kl_div(
                F.log_softmax(out_v_masked / temperature, dim=1),
                teacher_prob,
                reduction='batchmean'
            ) * (temperature ** 2)
            visual_kd_specialist = F.kl_div(
                F.log_softmax(specialist_visual / temperature, dim=1),
                teacher_prob,
                reduction='batchmean'
            ) * (temperature ** 2)
            visual_kd = 0.5 * visual_kd_shared + 0.5 * visual_kd_specialist
            loss = loss + weight_visual * args.sr_kd_weight * visual_kd

    stats = {
        'sr_loss': float(loss.detach().item()),
        'sr_weight_audio': weight_audio,
        'sr_weight_visual': weight_visual,
        'sr_gap_audio': float(repair_weights['gap_audio']),
        'sr_gap_visual': float(repair_weights['gap_visual']),
        'sr_audio_shared_ce': float(audio_shared_ce.detach().item()),
        'sr_visual_shared_ce': float(visual_shared_ce.detach().item()),
        'sr_audio_specialist_ce': float(audio_specialist_ce.detach().item()),
        'sr_visual_specialist_ce': float(visual_specialist_ce.detach().item()),
        'sr_audio_fused_ce': float(audio_fused_ce.detach().item()),
        'sr_visual_fused_ce': float(visual_fused_ce.detach().item()),
        'sr_audio_kd': float(audio_kd.detach().item()),
        'sr_visual_kd': float(visual_kd.detach().item()),
        'sr_teacher_conf': float(teacher_conf),
    }
    return loss, stats


def freeze_head(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.head.parameters():
        param.requires_grad = False

def open_head(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.head.parameters():
        param.requires_grad = True




def open_audio_net(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.audio_net.parameters():
        param.requires_grad = True

def freeze_visual_net(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.visual_net.parameters():
        param.requires_grad = False
        

def open_visual_net(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.visual_net.parameters():
        param.requires_grad = True




def calculate_weight_mean(network):
    total_sum = 0
    total_count = 0

    for param in network.parameters():
        if param.requires_grad:
            total_sum += param.data.sum().item()
            total_count += param.data.numel()

    # 计算并返回权重的平均值
    weight_mean = total_sum / total_count if total_count > 0 else 0
    return weight_mean

def calculate_audio_visual_weight_mean(model):
    # 如果使用 DataParallel，获取原始模型
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # 计算 audio_net 的权重平均值
    audio_mean = calculate_weight_mean(model.audio_net)

    # 计算 visual_net 的权重平均值
    visual_mean = calculate_weight_mean(model.visual_net)

    return audio_mean, visual_mean


def calculate_proximal_term(model, global_model , epoch):
    proximal_term = 0.0

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if isinstance(global_model, torch.nn.DataParallel):
        global_model = global_model.module

    for w, w_t in zip(model.audio_net.parameters(), global_model.audio_net.parameters()):
        proximal_term += (w - w_t).norm(2)  
        if(epoch < 10):
            return (1/2) * proximal_term
        elif(epoch < 20):
            return (1/2) * proximal_term
        elif(epoch < 30):
            return (1/2) * proximal_term
        else:
            return (1/2) * proximal_term


def calculate_proximal_term2(model, global_model , epoch):
    proximal_term = 0.0

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if isinstance(global_model, torch.nn.DataParallel):
        global_model = global_model.module
    for w, w_t in zip(model.visual_net.parameters(), global_model.visual_net.parameters()):
        proximal_term += (w_t - w).norm(2) 
        if(epoch < 10):
            return (5/2) * proximal_term
        elif(epoch < 20):
            return (5/2) * proximal_term
        elif(epoch < 30):
            return (1/2) * proximal_term
        else:
            return (1/2) * proximal_term



def log_images_to_tensorboard(writer, dataloader, device, phase='train', num_images=16):
    # Get a batch of images and labels
    spec, images, labels = next(iter(dataloader))
    
    # Move images to the device (if using GPU)
    images = images.to(device)
    # Squeeze the extra dimension (batch_size, 3, 1, 224, 224) -> (batch_size, 3, 224, 224)
    images = images.squeeze(2)
    
    # Select the first `num_images` images and labels to log
    img_grid = torchvision.utils.make_grid(images[:num_images])
    
    # Add images to TensorBoard
    writer.add_image(f'{phase}_images', img_grid)
    
    # If you want to add labels, make sure they are in the proper format (e.g., string)
    # Here we'll just log labels as scalars (for simplicity)
    for i in range(min(num_images, len(labels))):
        writer.add_text(f'{phase}_label_{i}', str(labels[i].item()), global_step=0)


def log_images_to_tensorboard(writer, dataloader, device, phase='train'):
    for step, (spec, images, labels) in enumerate(dataloader):
        # Move images to the device (if using GPU)
        images = images.to(device)
        
        # Squeeze the extra dimension (batch_size, 3, 1, 224, 224) -> (batch_size, 3, 224, 224)
        images = images.squeeze(2)
        
        # Log each image and its label separately
        for i in range(len(images)):
            # Add each individual image to TensorBoard
            writer.add_image(f'{phase}_image_{step}_{i}', images[i], global_step=step)
            
            # Add the corresponding label as text
            writer.add_text(f'{phase}_label_{step}_{i}', str(labels[i].item()), global_step=step)
        
        # Optionally, break the loop after a few steps if you don't want to log the whole dataset
        if step == 10:
            break







def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None, visual_trace_list=None, audio_trace_list=None, epoch_data_lists=None, lrmo_state=None):
    cmob_tool = CMoB(device) if args.use_cmob else None
    total_audio_grad_sum = 0
    total_visual_grad_sum = 0
    total_audio_count = 0
    total_visual_count = 0
    alpha1 = 0 
    k_audio = compute_trace_progress(audio_trace_list, epoch)
    k_visual = compute_trace_progress(visual_trace_list, epoch)
    if args.use_dual_rescue:
        k = max(k_audio, k_visual)
    else:
        k = k_audio
        


    
    
    if epoch < 0:
        mu_v = 1
        mu_a = 2.5
    else:
        mu_a =0
    
    ce_criterion = nn.CrossEntropyLoss()
    criterion = ce_criterion
    focal_criterion = None
    current_gamma = 0.0
    sw_enabled = False
    if args.use_sample_weighting:
        if args.sw_mode == 'always':
            sw_enabled = True
        elif args.sw_mode == 'epoch_range':
            sw_enabled = (epoch >= args.sw_start_epoch) and (epoch <= args.sw_end_epoch)
        else:
            sw_enabled = (max(k_audio, k_visual) > args.sw_k_threshold) if args.use_dual_rescue else (k > args.sw_k_threshold)

        if sw_enabled:
            progress = min(1.0, epoch / max(1, args.lr_decay_step))
            current_gamma = args.focal_gamma_start + (args.focal_gamma_end - args.focal_gamma_start) * progress
            focal_criterion = FocalLoss(gamma=current_gamma)
    
    if args.use_contrastive:
        contrastive_criterion = CrossModalContrastiveLoss()
    
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    lrmo_state = get_lrmo_state(lrmo_state)
    global_model = cp.deepcopy(model)
    visual_cos_similarity = 0
    audio_cos_similarity = 0
    mm_cos_similarity = 0
    fisher_matrix = None
    # optimizer_audio = optim.SGD(model.module.audio_net.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # scheduler_audio = optim.lr_scheduler.StepLR(optimizer_audio, args.lr_decay_step, args.lr_decay_ratio)

    model.train()
    print("Start training ... ")

    record_names_audio = []
    record_names_visual = []
    for name, param in model.named_parameters():
        if 'head' in name: 
            continue
        if ('audio' in name):
            record_names_audio.append((name, param))
            continue
        if ('visual' in name):
            record_names_visual.append((name, param))
            continue

    _loss_con = 0
    _loss_a = 0
    _loss_v = 0
    
    # 确保 _loss 被初始化
    _loss = 0



    


    
    fim = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if module.weight.requires_grad:
                fim[name] = torch.zeros_like(module.weight)
    fim_audio = {}
    fim_visual = {}
    fim_audio_head = {}
    fim_visual_head = {}
    fim_mm_head = {}


    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if 'audio_net' in name:
                if module.weight.requires_grad:
                    fim_audio[name] = torch.zeros_like(module.weight)
            elif 'visual_net' in name:
                if module.weight.requires_grad:
                    fim_visual[name] = torch.zeros_like(module.weight)
            elif 'head_audio' in name:
                if module.weight.requires_grad:
                    fim_audio_head[name] = torch.zeros_like(module.weight)
            elif 'head_video' in name:
                if module.weight.requires_grad:
                    fim_visual_head[name] = torch.zeros_like(module.weight)
            elif 'head' in name:
                if module.weight.requires_grad:
                    fim_mm_head[name] = torch.zeros_like(module.weight)
            

    
    w_a,w_v = calculate_audio_visual_weight_mean(model)
    
    writer.add_scalar('w_a',w_a,epoch)
    writer.add_scalar('w_v',w_v,epoch)
    writer.add_scalar('k_audio', k_audio, epoch)
    writer.add_scalar('k_visual', k_visual, epoch)
    writer.add_scalar('k_joint_gate', k, epoch)

    weights_head = model.module.head.weight.data
    weights_audio_head = model.module.head_audio.weight.data
    weights_video_head = model.module.head_video.weight.data
    cos_similarity_audio_head = F.cosine_similarity(
        weights_audio_head, weights_head[:, :512], dim=1).mean().item()

    cos_similarity_video_head = F.cosine_similarity(
        weights_video_head, weights_head[:, 512:], dim=1).mean().item()
    cos_similarity_head = F.cosine_similarity(
        weights_head[:, :512], weights_head[:, 512:], dim=1).mean().item()

    writer.add_scalar('cos_similarity_audio_head',cos_similarity_audio_head,epoch)
    writer.add_scalar('cos_similarity_video_head',cos_similarity_video_head,epoch)
    writer.add_scalar('cos_similarity_head',cos_similarity_head,epoch)
    

    # fim = compute_fim(model, criterion, dataloader, device)
    # log_images_to_tensorboard(writer, dataloader, device, phase='train', num_images=16)


    for step, (spec, image, label) in enumerate(dataloader):

        #pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        lr = optimizer.param_groups[0]['lr']

        # TODO: make it simpler and easier to extend
        a, v, a_output, v_output, out, proj_a, proj_v = model(spec.unsqueeze(1).float(), image.float())
        out_a, out_v = get_unimodal_logits(args, model, a, v, a_output, v_output, apply_aux_blend=False)

        with torch.no_grad():
            audio_prob_train = softmax(out_a)
            visual_prob_train = softmax(out_v)
            conf_audio_batch = torch.mean(audio_prob_train.gather(1, label.view(-1, 1))).item()
            conf_visual_batch = torch.mean(visual_prob_train.gather(1, label.view(-1, 1))).item()

        lrmo_stats = None
        if args.use_lrmo:
            lrmo_stats = compute_lrmo_update_ratios(args, lrmo_state, conf_audio_batch, conf_visual_batch)
            if writer is not None:
                writer.add_scalar('lrmo/conf_audio', conf_audio_batch, epoch)
                writer.add_scalar('lrmo/conf_visual', conf_visual_batch, epoch)

        # if args.fusion_method == 'sum':
        #     out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
        #              model.module.fusion_module.fc_y.bias)
        #     out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
        #              model.module.fusion_module.fc_x.bias)
        # else:
        #     weight_size = model.module.fusion_module.fc_out.weight.size(1)
        #     out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
        #              + model.module.fusion_module.fc_out.bias / 2)

        #     out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
        #              + model.module.fusion_module.fc_out.bias / 2)

        if args.use_sample_weighting and sw_enabled and focal_criterion is not None:
            loss = focal_criterion(out, label)
        else:
            loss = ce_criterion(out, label)
        if args.use_contrastive:
            # Dynamic Contrastive Weight Schedule
            # Late stage: turn off contrastive loss to let classifier fine-tune
            current_con_weight = args.con_weight if epoch < args.lr_decay_step else 0.0
            
            if current_con_weight > 0 and proj_a is not None and proj_v is not None:
                loss_con = contrastive_criterion(proj_a, proj_v)
                loss = loss + current_con_weight * loss_con
            else:
                loss_con = torch.tensor(0.0).to(device)
        else:
            loss_con = torch.tensor(0.0).to(device)

        mag_balance_loss = torch.tensor(0.0).to(device)
        mag_compensation_loss = torch.tensor(0.0).to(device)
        if getattr(args, 'use_mag', False):
            model_module = get_model_module(model)
            mag_weights = getattr(model_module, 'latest_mag_weights', None)
            if mag_weights is not None:
                mag_audio_mean = mag_weights[:, 0].mean()
                mag_visual_mean = mag_weights[:, 1].mean()
                if writer is not None:
                    writer.add_scalar('mag/gate_audio', mag_audio_mean.item(), epoch)
                    writer.add_scalar('mag/gate_visual', mag_visual_mean.item(), epoch)
                if args.mag_balance_weight > 0 and epoch >= args.mag_balance_start_epoch:
                    gate_target = mag_weights.new_tensor([0.5, 0.5])
                    mag_balance_loss = F.mse_loss(mag_weights.mean(dim=0), gate_target)
                    loss = loss + args.mag_balance_weight * mag_balance_loss
                    if writer is not None:
                        writer.add_scalar('mag/balance_loss', mag_balance_loss.item(), epoch)
                if args.mag_compensation_weight > 0 and epoch >= args.mag_compensation_start_epoch:
                    conf_gap = abs(conf_audio_batch - conf_visual_batch)
                    if conf_gap > args.mag_compensation_margin:
                        inv_conf = mag_weights.new_tensor([
                            1.0 - conf_audio_batch,
                            1.0 - conf_visual_batch,
                        ])
                        target_temperature = max(args.mag_compensation_temperature, 1e-6)
                        gate_target = F.softmax(inv_conf / target_temperature, dim=0)
                    else:
                        gate_target = mag_weights.new_tensor([0.5, 0.5])

                    gate_mean = mag_weights.mean(dim=0)
                    mag_compensation_loss = F.mse_loss(gate_mean, gate_target)
                    loss = loss + args.mag_compensation_weight * mag_compensation_loss
                    if writer is not None:
                        writer.add_scalar('mag/target_audio', gate_target[0].item(), epoch)
                        writer.add_scalar('mag/target_visual', gate_target[1].item(), epoch)
                        writer.add_scalar('mag/compensation_loss', mag_compensation_loss.item(), epoch)

        if args.use_branch_calibration:
            bc_loss, bc_stats = compute_branch_calibration_loss(
                args=args,
                epoch=epoch,
                model=model,
                a=a,
                v=v,
                out=out,
                out_a_shared=out_a,
                out_v_shared=out_v,
                label=label,
            )
            loss = loss + bc_loss
            if bc_stats is not None:
                writer.add_scalar('bc_loss', bc_stats['bc_loss'], epoch)
                writer.add_scalar('bc_weight_audio', bc_stats['bc_weight_audio'], epoch)
                writer.add_scalar('bc_weight_visual', bc_stats['bc_weight_visual'], epoch)
                writer.add_scalar('bc_teacher_conf', bc_stats['bc_teacher_conf'], epoch)
                writer.add_scalar('bc_shared_conf_audio', bc_stats['bc_shared_conf_audio'], epoch)
                writer.add_scalar('bc_shared_conf_visual', bc_stats['bc_shared_conf_visual'], epoch)
                writer.add_scalar('bc_boosted_conf_audio', bc_stats['bc_boosted_conf_audio'], epoch)
                writer.add_scalar('bc_boosted_conf_visual', bc_stats['bc_boosted_conf_visual'], epoch)

        rescue_stats = None
        if args.use_dual_rescue:
            rescue_loss, rescue_stats = compute_dual_rescue_loss(
                args=args,
                epoch=epoch,
                out=out,
                out_audio_aux=a_output,
                out_visual_aux=v_output,
                label=label,
                k_audio=k_audio,
                k_visual=k_visual,
                device=device,
            )
            loss = loss + rescue_loss
            writer.add_scalar('rescue_loss', rescue_stats['rescue_loss'], epoch)
            writer.add_scalar('rescue_ce', rescue_stats['rescue_ce'], epoch)
            writer.add_scalar('rescue_kd', rescue_stats['rescue_kd'], epoch)
            writer.add_scalar('rescue_weight_audio', rescue_stats['rescue_weight_audio'], epoch)
            writer.add_scalar('rescue_weight_visual', rescue_stats['rescue_weight_visual'], epoch)
            writer.add_scalar('rescue_conf_audio', rescue_stats['conf_audio'], epoch)
            writer.add_scalar('rescue_conf_visual', rescue_stats['conf_visual'], epoch)

        specialist_stats = None
        # ===== KS 相对 baseline（3）的关键新增 =====
        # KS 当前成功版会在这里叠加一小份 specialist loss，
        # 目的是轻量补单模态能力，而不是把 baseline 主损失完全改掉。
        if getattr(args, 'use_branch_specialist', False):
            specialist_loss, specialist_stats = compute_branch_specialist_loss(
                args=args,
                epoch=epoch,
                model=model,
                a=a,
                v=v,
                out=out,
                out_a_shared=out_a,
                out_v_shared=out_v,
                out_audio_aux=a_output,
                out_visual_aux=v_output,
                label=label,
            )
            loss = loss + args.wbsr_loss_scale * specialist_loss
            if specialist_stats is not None and writer is not None:
                writer.add_scalar('bs/loss', specialist_stats['bs_loss'], epoch)
                writer.add_scalar('bs/audio_shared_ce', specialist_stats['bs_audio_shared_ce'], epoch)
                writer.add_scalar('bs/visual_shared_ce', specialist_stats['bs_visual_shared_ce'], epoch)
                writer.add_scalar('bs/audio_specialist_ce', specialist_stats['bs_audio_specialist_ce'], epoch)
                writer.add_scalar('bs/visual_specialist_ce', specialist_stats['bs_visual_specialist_ce'], epoch)
                writer.add_scalar('bs/kd', specialist_stats['bs_kd'], epoch)
                writer.add_scalar('bs/teacher_conf', specialist_stats['bs_teacher_conf'], epoch)

        repair_stats = None
        if getattr(args, 'use_single_run_repair', False):
            repair_loss, repair_stats = compute_single_run_repair_loss(
                args=args,
                epoch=epoch,
                model=model,
                spec=spec,
                image=image,
                out=out,
                label=label,
                conf_audio_batch=conf_audio_batch,
                conf_visual_batch=conf_visual_batch,
            )
            loss = loss + repair_loss
            if repair_stats is not None and writer is not None:
                writer.add_scalar('sr/loss', repair_stats['sr_loss'], epoch)
                writer.add_scalar('sr/weight_audio', repair_stats['sr_weight_audio'], epoch)
                writer.add_scalar('sr/weight_visual', repair_stats['sr_weight_visual'], epoch)
                writer.add_scalar('sr/gap_audio', repair_stats['sr_gap_audio'], epoch)
                writer.add_scalar('sr/gap_visual', repair_stats['sr_gap_visual'], epoch)
                writer.add_scalar('sr/audio_shared_ce', repair_stats['sr_audio_shared_ce'], epoch)
                writer.add_scalar('sr/visual_shared_ce', repair_stats['sr_visual_shared_ce'], epoch)
                writer.add_scalar('sr/audio_specialist_ce', repair_stats['sr_audio_specialist_ce'], epoch)
                writer.add_scalar('sr/visual_specialist_ce', repair_stats['sr_visual_specialist_ce'], epoch)
                writer.add_scalar('sr/audio_fused_ce', repair_stats['sr_audio_fused_ce'], epoch)
                writer.add_scalar('sr/visual_fused_ce', repair_stats['sr_visual_fused_ce'], epoch)
                writer.add_scalar('sr/audio_kd', repair_stats['sr_audio_kd'], epoch)
                writer.add_scalar('sr/visual_kd', repair_stats['sr_visual_kd'], epoch)
                writer.add_scalar('sr/teacher_conf', repair_stats['sr_teacher_conf'], epoch)

        loss_out_v = ce_criterion(out_v, label)
        loss_out_a = ce_criterion(out_a, label)

        if epoch_data_lists is not None:
            with torch.no_grad():
                _, predicted_labels_v = torch.max(visual_prob_train, dim=1)
                batch_acc_v = torch.mean((predicted_labels_v == label).float()).item()
                epoch_data_lists['conf_v'].append(conf_visual_batch)
                epoch_data_lists['acc_v'].append(batch_acc_v)

                _, predicted_labels_a = torch.max(audio_prob_train, dim=1)
                batch_acc_a = torch.mean((predicted_labels_a == label).float()).item()
                epoch_data_lists['conf_a'].append(conf_audio_batch)
                epoch_data_lists['acc_a'].append(batch_acc_a)


        losses=[loss,loss_out_a,loss_out_v]
        all_loss = ['both', 'audio', 'visual']
        grads_audio = {}
        grads_visual={}

        for idx, loss_type in enumerate(all_loss):
            loss_tem = losses[idx]
            loss_tem.backward(retain_graph=True)
            if(loss_type=='visual'):
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    if param.grad is not None:
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone() 
                    else:
                        grads_visual[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_visual])           
                average_grad_visual = torch.mean(grads_visual[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_visual',average_grad_visual,epoch)
            elif(loss_type=='audio'):
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    if param.grad is not None:
                        grads_audio[loss_type][tensor_name] = param.grad.data.clone() 
                    else:
                        grads_audio[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_audio])
                average_grad_audio = torch.mean(grads_audio[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_audio',average_grad_audio,epoch)
            else:
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    if param.grad is not None:
                        grads_audio[loss_type][tensor_name] = param.grad.data.clone() 
                    else:
                        grads_audio[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])
                average_grad_audio_mm = torch.mean(grads_audio[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_audio_mm',average_grad_audio_mm,epoch)
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    if param.grad is not None:
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                    else:
                        grads_visual[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])
                average_grad_visual_mm = torch.mean(grads_visual[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_visual_mm',average_grad_visual_mm,epoch)
            
            optimizer.zero_grad()


        loss_out_v.backward(retain_graph=True)  
        loss_out_a.backward(retain_graph=True) 
        loss.backward()

        # 初始化 beta
        beta = 0
        beta2 = 0
        model.usegate = False

        if args.use_cmob:
            # === CMDR paper module (legacy flag name: --use_cmob) ===
            # NOT per-sample CMoB valuation: calculate_causal_gap uses
            # CrossEntropyLoss(mean), so gap/TE are mini-batch scalar
            # dominance estimates for regulating encoder updates.
            gap, te_a, te_v = cmob_tool.calculate_causal_gap(model, spec, image, label, criterion)

            # 使用因果 Gap 来计算 Beta
            audio_window_open = k_audio > 0.05 if args.use_dual_rescue else k > 0.05
            visual_window_open = k_visual > 0.05 if args.use_dual_rescue else k > 0.05

            if gap > 0 and audio_window_open:  # Audio 强 -> 抑制 Audio
                beta = args.cmdr_beta_scale * torch.exp(gap)
            elif gap < 0 and visual_window_open:  # Visual 强 -> 抑制 Visual
                beta2 = args.cmdr_beta_scale * torch.exp(-gap)  # 注意这里取反确保指数为正

        else:
            # === 原有的基于 Score 的逻辑 ===
            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])

            audio_window_open = k_audio > 0.05 if args.use_dual_rescue else k > 0.05
            visual_window_open = k_visual > 0.05 if args.use_dual_rescue else k > 0.05

            if (score_a > score_v) and audio_window_open:
                gap = tanh(score_a - score_v)
                beta = args.cmdr_beta_scale * torch.exp(gap)
            elif (score_a < score_v) and visual_window_open:
                gap = tanh(score_v - score_a)
                beta2 = 0.1 * torch.exp(gap)

            
            
        for model_param, global_model_param in zip(model.parameters(),global_model.parameters()):
               
            if model_param.requires_grad and any(model_param is param for param in model.module.audio_net.parameters()):
                model_param.grad += beta * (model_param - global_model_param)

            if model_param.requires_grad and any(model_param is param for param in model.module.visual_net.parameters()): 
                model_param.grad += beta2 * (model_param - global_model_param)

        lrmo_mask_stats = None
        if args.use_lrmo and lrmo_stats is not None:
            lrmo_importance_cache = {}
            lrmo_importance_cache.update(fim_audio)
            lrmo_importance_cache.update(fim_visual)
            if args.lrmo_include_heads:
                lrmo_importance_cache.update(fim_audio_head)
                lrmo_importance_cache.update(fim_visual_head)
            lrmo_mask_stats = apply_lrmo_gradient_mask(
                args=args,
                model=model,
                epoch=epoch,
                lrmo_stats=lrmo_stats,
                writer=writer,
                importance_cache=lrmo_importance_cache,
            )

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                # 修改：增加 and module.weight.grad is not None
                if module.weight.requires_grad and module.weight.grad is not None:
                    fim[name] += (module.weight.grad * module.weight.grad)
                    fim[name].detach_()

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 修改点：每个分支都必须加 and module.weight.grad is not None
                if 'audio_net' in name and module.weight.requires_grad and module.weight.grad is not None:
                    fim_audio[name] += (module.weight.grad * module.weight.grad)
                    fim_audio[name].detach_()
                elif 'visual_net' in name and module.weight.requires_grad and module.weight.grad is not None:
                    fim_visual[name] += (module.weight.grad * module.weight.grad)
                    fim_visual[name].detach_()
                elif 'head_audio' in name and module.weight.requires_grad and module.weight.grad is not None:
                    fim_audio_head[name] += (module.weight.grad * module.weight.grad)
                    fim_audio_head[name].detach_()
                elif 'head_video' in name and module.weight.requires_grad and module.weight.grad is not None:
                    fim_visual_head[name] += (module.weight.grad * module.weight.grad)
                    fim_visual_head[name].detach_()
                elif 'head' in name and module.weight.requires_grad and module.weight.grad is not None:
                    fim_mm_head[name] += (module.weight.grad * module.weight.grad)
                    fim_mm_head[name].detach_()

        for name, parms in model.named_parameters():
            if parms.grad is not None: 
                layer = str(name).split('.')[1]

                if 'audio' in layer and len(parms.grad.size()) == 4:

                    audio_L2_norm_square = torch.sum(parms.grad ** 2).item()
                    total_audio_grad_sum += lr * audio_L2_norm_square
                    total_audio_count += 1


                if 'visual' in layer and len(parms.grad.size()) == 4:

                    visual_L2_norm_square = torch.sum(parms.grad ** 2).item()
                    total_visual_grad_sum += lr * visual_L2_norm_square
                    total_visual_count += 1


        optimizer.step()

        _loss += loss.item()
        if args.use_contrastive:
            _loss_con += loss_con.item()
        
        _loss_a += loss_out_a.item()
        _loss_v += loss_out_v.item()

    if total_audio_count > 0:
        epoch_audio_L2_norm_mean = total_audio_grad_sum / total_audio_count
    else:
        epoch_audio_L2_norm_mean = 0

    if total_visual_count > 0:
        epoch_visual_L2_norm_mean = total_visual_grad_sum / total_visual_count
    else:
        epoch_visual_L2_norm_mean = 0


    fim_trace = 0
    for name in fim:
        fim[name] = fim[name].mean().item()
        fim_trace += fim[name]

    fim_trace_audio = 0
    for name in fim_audio:
        fim_audio[name] = fim_audio[name].mean().item()
        fim_trace_audio += fim_audio[name]

    fim_trace_visual = 0
    for name in fim_visual:
        fim_visual[name] = fim_visual[name].mean().item()
        fim_trace_visual += fim_visual[name]

    fim_trace_visual_head = 0
    for name in fim_visual_head:
        fim_visual_head[name] = fim_visual_head[name].mean().item()
        fim_trace_visual_head += fim_visual_head[name]

    fim_trace_audio_head = 0
    for name in fim_audio_head:
        fim_audio_head[name] = fim_audio_head[name].mean().item()
        fim_trace_audio_head += fim_audio_head[name]

    fim_trace_mm_head = 0
    for name in fim_mm_head:
        fim_mm_head[name] = fim_mm_head[name].mean().item()
        fim_trace_mm_head += fim_mm_head[name]
    
    visual_trace_list.append(fim_trace_visual)
    audio_trace_list.append(fim_trace_audio)
    print(audio_trace_list)
    writer.add_scalar('fim_trace_visual_head',fim_trace_visual_head,epoch)
    writer.add_scalar('fim_trace_audio_head',fim_trace_audio_head,epoch)
    writer.add_scalar('fim_trace_mm_head',fim_trace_mm_head,epoch)

    scheduler.step()


    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _loss_con / len(dataloader), epoch_audio_L2_norm_mean, epoch_visual_L2_norm_mean,fim_trace,fim_trace_audio,fim_trace_visual, average_grad_visual, average_grad_audio, average_grad_visual_mm, average_grad_audio_mm


def valid(args, model, device, dataloader, epoch=None):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            spec, image = apply_input_mask(spec, image, args.input_mask_mode)

            # ===== KS 相对 baseline（3）的关键新增 =====
            # baseline（3）这里是单视角验证；当前版在 valid 中显式走多视角 forward，
            # 因此验证出的 Visual Acc 更稳定，也更符合我们后面保存 KS 结果的口径。
            out, out_a, out_v = forward_eval_logits(args, model, spec, image, epoch=epoch)

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                #pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    args = apply_ks_single_run_recipe(args)
    print(args)

    if getattr(args, 'run_cremad_pipeline', False):
        pipeline_cmd = [
            sys.executable, '-u', 'run_cremad_pipeline.py',
            '--gpu_ids', args.gpu_ids,
            '--random_seed', str(args.random_seed),
            '--base_epochs', str(args.pipeline_base_epochs),
            '--eval_num_views', str(args.pipeline_eval_num_views),
        ]
        if args.pipeline_run_name:
            pipeline_cmd.extend(['--run_name', args.pipeline_run_name])
        print('[Pipeline] delegating to run_cremad_pipeline.py', flush=True)
        raise SystemExit(subprocess.call(pipeline_cmd, cwd=os.path.dirname(os.path.abspath(__file__))))
    if getattr(args, 'run_ks_pipeline', False):
        pipeline_cmd = [
            sys.executable, '-u', 'run_ks_pipeline.py',
            '--gpu_ids', args.gpu_ids,
            '--random_seed', str(args.random_seed),
            '--base_ckpt', args.ks_pipeline_base_ckpt,
            '--audio_epochs', str(args.ks_pipeline_audio_epochs),
            '--visual_epochs', str(args.ks_pipeline_visual_epochs),
            '--joint_epochs', str(args.ks_pipeline_joint_epochs),
            '--pipeline_profile', args.ks_pipeline_profile,
        ]
        if args.pipeline_run_name:
            pipeline_cmd.extend(['--run_name', args.pipeline_run_name])
        print('[Pipeline] delegating to run_ks_pipeline.py', flush=True)
        raise SystemExit(subprocess.call(pipeline_cmd, cwd=os.path.dirname(os.path.abspath(__file__))))
    loss_list = []
    loss_a_list = []
    loss_v_list = []

    audio_NormList = [0,0,0,0,0,0,0,0,0,0]
    visual_NormList = [0,0,0,0,0,0,0,0,0,0]
    audio_FGNList = []
    visual_FGNList = []
    audio_GNorm_list = []
    visual_GNorm_list = []
    fim_list = []
    audio_fim_list = []
    visual_fim_list = []
    accuracy_list = []
    accuracy_list_audio = []
    accuracy_list_visual = []
    avarage_gradient_visual_mm_list = []
    avarage_gradient_audio_mm_list = []
    avarage_gradient_visual_list = []
    avarage_gradient_audio_list = []
    
    visual_trace_list = [0,0,0,0,0,0,0,0,0,0]
    audio_trace_list = [0,0,0,0,0,0,0,0,0,0]



    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()


    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'KineticSound':
        train_dataset = KS_dataset(args=args, mode='train', select_ratio=1, v_norm=True, a_norm=False, name="KS")
        test_dataset = KS_dataset(args=args, mode='test', select_ratio=1, v_norm=True, a_norm=False, name="KS")
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    print(f"Number of samples in training dataset: {len(train_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)
    goal_targets = resolve_triplet_goal_targets(args)
    if goal_targets is not None:
        print('[Goal] active thresholds -> Acc>{:.4f}, Audio Acc>{:.4f}, Visual Acc>{:.4f}'.format(
            goal_targets['acc'],
            goal_targets['audio_acc'],
            goal_targets['visual_acc'],
        ))
    # writer_path = os.path.join(args.tensorboard_path, args.dataset)
    # if not os.path.exists(writer_path):
    #     os.mkdir(writer_path)
    # log_name = '{}_{}'.format(args.fusion_method, args.modulation)
    # writer = SummaryWriter(os.path.join(writer_path, log_name))
    # log_images_to_tensorboard(writer, train_dataloader, device, phase='train')

    if args.train and args.stage2_visual_tune:
        if args.stage2_init_ckpt == '':
            raise ValueError('stage2_visual_tune requires --stage2_init_ckpt')

        print('Starting stage-2 {} tuning from {}'.format(args.stage2_target, args.stage2_init_ckpt))
        init_ckpt = load_checkpoint_into_model(model, args.stage2_init_ckpt, device, strict=False)
        stage2_reference_acc = None
        if isinstance(init_ckpt, dict) and 'acc' in init_ckpt:
            stage2_reference_acc = float(init_ckpt['acc'])

        teacher_model = AVClassifier(args)
        teacher_model.apply(weight_init)
        teacher_model.to(device)
        teacher_model = torch.nn.DataParallel(teacher_model, device_ids=gpu_ids)
        teacher_model.cuda()
        load_checkpoint_into_model(teacher_model, args.stage2_init_ckpt, device, strict=False)
        freeze_module_parameters(get_model_module(teacher_model))
        teacher_model.eval()
        audio_teacher_model = build_optional_teacher_model(args, args.stage2_audio_teacher_ckpt, device, gpu_ids, label='audio teacher')
        visual_teacher_model = build_optional_teacher_model(args, args.stage2_visual_teacher_ckpt, device, gpu_ids, label='visual teacher')

        prepare_stage2_student(model, args)
        optimizer = build_stage2_optimizer(args, model)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.stage2_lr_decay_step, args.stage2_lr_decay_ratio)

        writer = None
        if args.use_tensorboard:
            writer_path = os.path.join(args.tensorboard_path, args.dataset)
            os.makedirs(writer_path, exist_ok=True)
            log_name = '{}_{}_stage2_{}'.format(args.fusion_method, args.modulation, args.stage2_target)
            writer = SummaryWriter(os.path.join(writer_path, log_name))

        best_metric = -1.0
        best_acc = -1.0
        best_acc_a = -1.0
        best_acc_v = -1.0
        best_goal_candidate = None
        best_goal_qualified = None
        best_goal_save_path = ''
        stage2_acc_floor = None
        if stage2_reference_acc is not None and args.stage2_acc_drop_tolerance >= 0:
            stage2_acc_floor = max(0.0, stage2_reference_acc - args.stage2_acc_drop_tolerance)
            print('[Stage2] selection accuracy floor: {:.3f} (init acc {:.3f} - tol {:.3f})'.format(
                stage2_acc_floor,
                stage2_reference_acc,
                args.stage2_acc_drop_tolerance,
            ))

        stage2_display_best_acc = -1.0

        teacher_focus_metric = None
        if args.input_mask_mode == 'audio_only':
            teacher_focus_metric = 'Audio Acc'
        elif args.input_mask_mode == 'visual_only':
            teacher_focus_metric = 'Visual Acc'

        if teacher_focus_metric is not None:
            print('[Teacher] focus on {}; Acc and the other modality are reference only.'.format(
                teacher_focus_metric
            ))

        for epoch in range(args.stage2_epochs):
            print('Epoch: {}: '.format(epoch))
            print('Start training ... ')
            stage2_stats = stage2_visual_epoch(
                args=args,
                epoch=epoch,
                model=model,
                teacher_model=teacher_model,
                device=device,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                writer=writer,
                audio_teacher_model=audio_teacher_model,
                visual_teacher_model=visual_teacher_model,
            )
            acc, acc_a, acc_v = valid(args, model, device, test_dataloader, epoch=epoch)
            if args.stage2_save_metric == 'visual':
                metric_value = acc_v
            elif args.stage2_save_metric == 'audio':
                metric_value = acc_a
            elif args.stage2_save_metric == 'sum_unimodal':
                metric_value = acc_a + acc_v
            else:
                metric_value = acc

            if writer is not None:
                writer.add_scalars('stage2/eval', {
                    'acc': acc,
                    'acc_a': acc_a,
                    'acc_v': acc_v,
                }, epoch)

            goal_summary = evaluate_triplet_goal(acc, acc_a, acc_v, goal_targets)
            if goal_summary is not None:
                if writer is not None:
                    writer.add_scalars('goal/margin', {
                        'acc': goal_summary['margins']['acc'],
                        'audio_acc': goal_summary['margins']['audio_acc'],
                        'visual_acc': goal_summary['margins']['visual_acc'],
                    }, epoch)
                    writer.add_scalar('goal/min_margin', goal_summary['min_margin'], epoch)
                    writer.add_scalar('goal/sum_margin', goal_summary['sum_margin'], epoch)
                if is_better_triplet_goal(goal_summary, best_goal_candidate, args.goal_score_mode):
                    best_goal_candidate = dict(goal_summary)
                    best_goal_candidate['epoch'] = epoch
                print_triplet_goal_status(goal_summary, prefix='[Goal][Stage2]')
                if goal_summary['meets_all'] and is_better_triplet_goal(goal_summary, best_goal_qualified, args.goal_score_mode):
                    best_goal_qualified = dict(goal_summary)
                    best_goal_qualified['epoch'] = epoch
                    if args.save_goal_ckpt:
                        best_goal_save_path = save_goal_checkpoint(
                            args=args,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            acc=acc,
                            acc_a=acc_a,
                            acc_v=acc_v,
                            goal_summary=goal_summary,
                            stage_name='stage2_{}'.format(args.stage2_target),
                        )
                        print('[Goal][Stage2] checkpoint saved at {}.'.format(best_goal_save_path))

            stage2_display_best_acc = max(stage2_display_best_acc, float(acc))
            print('[Stage2] AudioCE: {:.3f} VisualCE: {:.3f} AudioSpecCE: {:.3f} VisualSpecCE: {:.3f} KD: {:.3f} Consistency: {:.3f}'.format(
                stage2_stats['audio_ce'],
                stage2_stats['visual_ce'],
                stage2_stats['audio_specialist_ce'],
                stage2_stats['visual_specialist_ce'],
                stage2_stats['kd'],
                stage2_stats['consistency'],
            ))
            if getattr(args, 'stage2_adaptive_repair', False) and args.stage2_target == 'both':
                print('[Stage2Adaptive] AudioW: {:.3f} VisualW: {:.3f} ConsScale: {:.3f}'.format(
                    stage2_stats['audio_weight'],
                    stage2_stats['visual_weight'],
                    stage2_stats['consistency_scale'],
                ))
            print('Epoch:  {}   Loss: {:.3f}  ConLoss: {:.3f}  Acc: {:.3f}  Audio Acc: {:.3f}  Visual Acc: {:.3f}  Best Acc: {:.3f}'.format(
                epoch,
                stage2_stats['loss'],
                0.0,
                acc,
                acc_a,
                acc_v,
                stage2_display_best_acc,
            ))

            meets_acc_floor = stage2_acc_floor is None or acc >= stage2_acc_floor
            if not meets_acc_floor:
                print('[Stage2] epoch {} skipped for model selection because Acc {:.3f} < floor {:.3f}'.format(
                    epoch, acc, stage2_acc_floor
                ))
            elif metric_value > best_metric:
                best_metric = float(metric_value)
                best_acc = float(acc)
                best_acc_a = float(acc_a)
                best_acc_v = float(acc_v)
                save_dir = save_stage2_checkpoint(args, model, optimizer, scheduler, epoch, acc, acc_a, acc_v, metric_value)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(stage2_stats['loss'], acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(stage2_stats['loss'], acc, stage2_display_best_acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))

        if writer is not None:
            writer.close()

        print('[Stage2] finished. Best {}: {:.3f} | Best Acc: {:.3f} | Audio Acc: {:.3f} | Visual Acc: {:.3f}'.format(
            args.stage2_save_metric,
            best_metric,
            best_acc,
            best_acc_a,
            best_acc_v,
        ))
        if goal_targets is not None:
            if best_goal_qualified is not None:
                summary_message = '[Goal][Stage2] best qualified epoch {} | Acc {:.3f} | Audio Acc {:.3f} | Visual Acc {:.3f}'.format(
                    best_goal_qualified['epoch'],
                    best_goal_qualified['acc'],
                    best_goal_qualified['audio_acc'],
                    best_goal_qualified['visual_acc'],
                )
                if best_goal_save_path:
                    summary_message += ' | saved at {}'.format(best_goal_save_path)
                print(summary_message)
            elif best_goal_candidate is not None:
                print('[Goal][Stage2] no epoch exceeded all targets; closest epoch {} | margins Acc {:+.4f} Audio {:+.4f} Visual {:+.4f}'.format(
                    best_goal_candidate['epoch'],
                    best_goal_candidate['margins']['acc'],
                    best_goal_candidate['margins']['audio_acc'],
                    best_goal_candidate['margins']['visual_acc'],
                ))
        return

    if args.train:

        best_acc = 0.0
        lrmo_state = get_lrmo_state(None) if args.use_lrmo else None
        best_goal_candidate = None
        best_goal_qualified = None
        best_goal_save_path = ''

        for epoch in range(args.epochs):
            total_audio_grad_sum = 0
            total_visual_grad_sum = 0
            total_audio_count = 0
            total_visual_count = 0
            # === 新增代码开始: 在每个 epoch 开始时，为置信度分析创建数据列表 ===
            epoch_data_lists = {
                'conf_v': [], 'acc_v': [],
                'conf_a': [], 'acc_a': [],
            }

            print('Epoch: {}: '.format(epoch))
            writer = None

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))
                
                
                batch_loss, batch_loss_a, batch_loss_v, batch_loss_con, epoch_audio_L2_norm_mean, epoch_visual_L2_norm_mean,fim_trace ,fim_trace_audio,fim_trace_visual,average_grad_visual,average_grad_audio,average_grad_visual_mm,average_grad_audio_mm = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler, writer, visual_trace_list, audio_trace_list, epoch_data_lists, lrmo_state)
                fim_list.append(fim_trace)
                loss_list.append(batch_loss)
                loss_a_list.append(batch_loss_a)
                loss_v_list.append(batch_loss_v)
                avarage_gradient_visual_list.append(average_grad_visual)
                avarage_gradient_audio_list.append(average_grad_audio)
                avarage_gradient_visual_mm_list.append(average_grad_visual_mm)
                avarage_gradient_audio_mm_list.append(average_grad_audio_mm)
                audio_fim_list.append(fim_trace_audio)
                visual_fim_list.append(fim_trace_visual)
                # visual_trace_list.append(fim_trace_visual)
                # audio_trace_list.append(fim_trace_audio)
                audio_GNorm_list.append(epoch_audio_L2_norm_mean)
                visual_GNorm_list.append(epoch_visual_L2_norm_mean)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader, epoch=epoch)
                accuracy_list.append(acc)
                accuracy_list_visual.append(acc_v)
                accuracy_list_audio.append(acc_a)
                audio_NormList.append(epoch_audio_L2_norm_mean)
                audio_OldNorm = max([np.mean(audio_NormList[-11:-1]), 0.0000001])
                audio_NewNorm = np.mean(audio_NormList[-11:])
                audio_FGNList.append((audio_NewNorm - audio_OldNorm) / audio_NewNorm)
                print("audio_FGN:", (audio_NewNorm - audio_OldNorm) / audio_OldNorm)
                visual_NormList.append(epoch_visual_L2_norm_mean)
                visual_OldNorm = max([np.mean(visual_NormList[-11:-1]), 0.0000001])
                visual_NewNorm = np.mean(visual_NormList[-11:])
                visual_FGNList.append((visual_NewNorm - visual_OldNorm) / visual_NewNorm)
                print("visual_FGN:", (visual_NewNorm - visual_OldNorm) / visual_OldNorm)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

                writer.add_scalar('FIM_Trace', fim_trace,epoch)
                writer.add_scalar('FIM_Trace_audio', fim_trace_audio,epoch)
                writer.add_scalar('Fim_Trace_visual',fim_trace_visual,epoch)
                # ==================== 最终版本的绘图代码 ===============================
                print(f"Epoch {epoch} 完成, 开始绘制 Batch 级置信度-准确率散点图...")
                
                plt.figure(figsize=(12, 12))
                
                # 绘制 y=x 对角线作为参考
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x (Reference Line)')
                
                # 从 epoch_data_lists 字典中取数据绘图
                plt.scatter(epoch_data_lists['conf_v'], epoch_data_lists['acc_v'], color='blue', alpha=0.6, s=30, label='Visual Modality (Batch Points)')
                plt.scatter(epoch_data_lists['conf_a'], epoch_data_lists['acc_a'], color='red', alpha=0.6, s=30, label='Audio Modality (Batch Points)')

                plt.title(f'Batch-Level Confidence vs. Accuracy - Epoch {epoch}', fontsize=16)
                plt.xlabel('Average Confidence per Batch', fontsize=12)
                plt.ylabel('Accuracy per Batch', fontsize=12)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.gca().set_aspect('equal', adjustable='box')

                plot_save_path = os.path.join('results', 'batch_scatter_plots')
                os.makedirs(plot_save_path, exist_ok=True)
                
                plt.savefig(os.path.join(plot_save_path, f'batch_scatter_epoch_{epoch}.png'))
                print(f"Batch级散点图已保存到: {os.path.join(plot_save_path, f'batch_scatter_epoch_{epoch}.png')}")
                plt.close()
                # ==================== 绘图代码结束 ========================================

                                            

            else:
                batch_loss, batch_loss_a, batch_loss_v, epoch_audio_L2_norm_mean, epoch_visual_L2_norm_mean,fim_trace,fim_trace_audio,fim_trace_visual = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler, epoch_data_lists=epoch_data_lists, lrmo_state=lrmo_state)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader, epoch=epoch)
                audio_GNorm_list.append(epoch_audio_L2_norm_mean)
                visual_GNorm_list.append(epoch_visual_L2_norm_mean)
                audio_NormList.append(epoch_audio_L2_norm_mean)
                audio_OldNorm = max([np.mean(audio_NormList[-2:-1]), 0.0000001])
                audio_NewNorm = np.mean(audio_NormList[-2:])
                audio_FGNList.append((audio_NewNorm - audio_OldNorm) / audio_NewNorm)
                print("audio_FGN:", (audio_NewNorm - audio_OldNorm) / audio_OldNorm)
                visual_NormList.append(epoch_visual_L2_norm_mean)
                visual_OldNorm = max([np.mean(visual_NormList[-2:-1]), 0.0000001])
                visual_NewNorm = np.mean(visual_NormList[-2:])
                visual_FGNList.append((visual_NewNorm - visual_OldNorm) / visual_NewNorm)
                print("visual_FGN:", (visual_NewNorm - visual_OldNorm) / visual_OldNorm)

            print('Epoch:  {}   Loss: {:.3f}  ConLoss: {:.3f}  Acc: {:.3f}  Audio Acc: {:.3f}  Visual Acc: {:.3f}  Best Acc: {:.3f}'.format(epoch, batch_loss, batch_loss_con, acc, acc_a, acc_v, max(best_acc, acc)))

            goal_summary = evaluate_triplet_goal(acc, acc_a, acc_v, goal_targets)
            if goal_summary is not None:
                if writer is not None:
                    writer.add_scalars('goal/margin', {
                        'acc': goal_summary['margins']['acc'],
                        'audio_acc': goal_summary['margins']['audio_acc'],
                        'visual_acc': goal_summary['margins']['visual_acc'],
                    }, epoch)
                    writer.add_scalar('goal/min_margin', goal_summary['min_margin'], epoch)
                    writer.add_scalar('goal/sum_margin', goal_summary['sum_margin'], epoch)
                if is_better_triplet_goal(goal_summary, best_goal_candidate, args.goal_score_mode):
                    best_goal_candidate = dict(goal_summary)
                    best_goal_candidate['epoch'] = epoch
                print_triplet_goal_status(goal_summary)
                if goal_summary['meets_all'] and is_better_triplet_goal(goal_summary, best_goal_qualified, args.goal_score_mode):
                    best_goal_qualified = dict(goal_summary)
                    best_goal_qualified['epoch'] = epoch
                    if args.save_goal_ckpt:
                        best_goal_save_path = save_goal_checkpoint(
                            args=args,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            acc=acc,
                            acc_a=acc_a,
                            acc_v=acc_v,
                            goal_summary=goal_summary,
                            stage_name='main',
                        )
                        print('[Goal] checkpoint saved at {}.'.format(best_goal_save_path))

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_of_dataset_{}_{}_alpha_{}_' \
                             'optimizer_{}_training_epochs_{}_' \
                             'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                          args.modulation,
                                                          args.alpha,
                                                          args.optimizer,
                                                          args.modulation_starts,
                                                          epoch, acc)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'acc_a': acc_a,
                              'acc_v': acc_v,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"experiments_{timestamp}_{args.modulation}"
        results_path = os.path.join(os.getcwd(), "results", experiment_folder)
        os.makedirs(results_path, exist_ok=True)

        data_to_save = {
            "fim_list": fim_list,
            "audio_fim_list": audio_fim_list,
            "visual_fim_list": visual_fim_list,
            "accuracy_list": accuracy_list,
            "accuracy_list_audio": accuracy_list_audio,
            "accuracy_list_visual": accuracy_list_visual,
            "avarage_gradient_visual_mm_list": avarage_gradient_visual_mm_list,
            "avarage_gradient_audio_mm_list": avarage_gradient_audio_mm_list,
            "avarage_gradient_visual_list": avarage_gradient_visual_list,
            "avarage_gradient_audio_list": avarage_gradient_audio_list,
            "loss_list":loss_list,
            "loss_a_list":loss_a_list,
            "loss_v_list":loss_v_list
            
        }

        for name, data in data_to_save.items():
            pkl_path = os.path.join(results_path, f"{name}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)
            csv_path = os.path.join(results_path, f"{name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                if isinstance(data[0], (list, tuple)):
                    writer.writerows(data)
                else:
                    writer.writerow(data)

        if goal_targets is not None:
            if best_goal_qualified is not None:
                summary_message = '[Goal] best qualified epoch {} | Acc {:.3f} | Audio Acc {:.3f} | Visual Acc {:.3f}'.format(
                    best_goal_qualified['epoch'],
                    best_goal_qualified['acc'],
                    best_goal_qualified['audio_acc'],
                    best_goal_qualified['visual_acc'],
                )
                if best_goal_save_path:
                    summary_message += ' | saved at {}'.format(best_goal_save_path)
                print(summary_message)
            elif best_goal_candidate is not None:
                print('[Goal] no epoch exceeded all targets; closest epoch {} | margins Acc {:+.4f} Audio {:+.4f} Visual {:+.4f}'.format(
                    best_goal_candidate['epoch'],
                    best_goal_candidate['margins']['acc'],
                    best_goal_candidate['margins']['audio_acc'],
                    best_goal_candidate['margins']['visual_acc'],
                ))


    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path, map_location=device)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        load_checkpoint_into_model(model, args.ckpt_path, device, strict=False)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))
        goal_summary = evaluate_triplet_goal(acc, acc_a, acc_v, goal_targets)
        print_triplet_goal_status(goal_summary)


if __name__ == "__main__":
    main()
