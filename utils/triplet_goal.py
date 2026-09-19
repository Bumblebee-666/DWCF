import os

import torch


PAPER_TRIPLET_TARGETS = {
    'CREMAD': {
        'acc': 0.7571,
        'audio_acc': 0.6163,
        'visual_acc': 0.6122,
    },
    'KineticSound': {
        'acc': 0.7203,
        'audio_acc': 0.5721,
        'visual_acc': 0.5357,
    },
}


def resolve_triplet_goal_targets(args):
    target_acc = float(getattr(args, 'target_acc', -1.0))
    target_audio_acc = float(getattr(args, 'target_audio_acc', -1.0))
    target_visual_acc = float(getattr(args, 'target_visual_acc', -1.0))

    if getattr(args, 'use_author_targets', False):
        preset = PAPER_TRIPLET_TARGETS.get(args.dataset)
        if preset is None:
            raise ValueError('No paper target preset is defined for dataset {}'.format(args.dataset))
        if target_acc < 0:
            target_acc = preset['acc']
        if target_audio_acc < 0:
            target_audio_acc = preset['audio_acc']
        if target_visual_acc < 0:
            target_visual_acc = preset['visual_acc']

    has_any_target = any(metric > 0 for metric in [target_acc, target_audio_acc, target_visual_acc])
    if not has_any_target:
        return None

    if not all(metric > 0 for metric in [target_acc, target_audio_acc, target_visual_acc]):
        raise ValueError('Triplet goal requires all three targets: --target_acc, --target_audio_acc, --target_visual_acc')

    return {
        'acc': target_acc,
        'audio_acc': target_audio_acc,
        'visual_acc': target_visual_acc,
    }


def evaluate_triplet_goal(acc, acc_a, acc_v, goal_targets):
    if goal_targets is None:
        return None

    margins = {
        'acc': float(acc - goal_targets['acc']),
        'audio_acc': float(acc_a - goal_targets['audio_acc']),
        'visual_acc': float(acc_v - goal_targets['visual_acc']),
    }

    return {
        'acc': float(acc),
        'audio_acc': float(acc_a),
        'visual_acc': float(acc_v),
        'targets': dict(goal_targets),
        'margins': margins,
        'meets_all': all(margin > 0.0 for margin in margins.values()),
        'min_margin': float(min(margins.values())),
        'sum_margin': float(sum(margins.values())),
    }


def is_better_triplet_goal(current_goal, best_goal, score_mode='min_margin'):
    if current_goal is None:
        return False
    if best_goal is None:
        return True

    if score_mode == 'sum_margin':
        current_key = (
            current_goal['sum_margin'],
            current_goal['min_margin'],
            current_goal['acc'],
            current_goal['audio_acc'],
            current_goal['visual_acc'],
        )
        best_key = (
            best_goal['sum_margin'],
            best_goal['min_margin'],
            best_goal['acc'],
            best_goal['audio_acc'],
            best_goal['visual_acc'],
        )
    else:
        current_key = (
            current_goal['min_margin'],
            current_goal['sum_margin'],
            current_goal['acc'],
            current_goal['audio_acc'],
            current_goal['visual_acc'],
        )
        best_key = (
            best_goal['min_margin'],
            best_goal['sum_margin'],
            best_goal['acc'],
            best_goal['audio_acc'],
            best_goal['visual_acc'],
        )

    return current_key > best_key


def save_goal_checkpoint(args, model, optimizer, scheduler, epoch, acc, acc_a, acc_v, goal_summary, stage_name='main'):
    os.makedirs(args.ckpt_path, exist_ok=True)
    safe_stage_name = str(stage_name).replace(' ', '_')
    model_name = '{}_{}_{}_epoch_{}_acc_{:.4f}_audio_{:.4f}_visual_{:.4f}.pth'.format(
        args.goal_ckpt_tag,
        safe_stage_name,
        args.dataset,
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
        'goal_score_mode': args.goal_score_mode,
        'goal_targets': goal_summary['targets'],
        'goal_margins': goal_summary['margins'],
        'goal_min_margin': goal_summary['min_margin'],
        'goal_sum_margin': goal_summary['sum_margin'],
        'goal_meets_all': goal_summary['meets_all'],
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(saved_dict, save_dir)
    return save_dir


def print_triplet_goal_status(goal_summary, prefix='[Goal]'):
    if goal_summary is None:
        return

    status = 'PASS' if goal_summary['meets_all'] else 'pending'
    print('{} status={} | margins Acc {:+.4f} Audio {:+.4f} Visual {:+.4f}'.format(
        prefix,
        status,
        goal_summary['margins']['acc'],
        goal_summary['margins']['audio_acc'],
        goal_summary['margins']['visual_acc'],
    ))
