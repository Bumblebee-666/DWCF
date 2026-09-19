import torch.nn as nn


def get_model_module(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def branch_specialist_is_active(args, epoch):
    return getattr(args, 'use_branch_specialist', False) and epoch is not None and epoch >= args.bs_start_epoch


def single_run_repair_is_active(args, epoch):
    return getattr(args, 'use_single_run_repair', False) and epoch is not None and args.sr_start_epoch <= epoch <= args.sr_end_epoch


def get_specialist_logits(model, a, v, audio_logits_aux=None, visual_logits_aux=None):
    module = get_model_module(model)
    if audio_logits_aux is None:
        audio_logits_aux = module.head_audio(a)
    if visual_logits_aux is None:
        visual_logits_aux = module.head_video(v)

    boost_audio = module.boost_head_audio(a)
    boost_visual = module.boost_head_video(v)
    specialist_audio = 0.5 * audio_logits_aux + 0.5 * boost_audio
    specialist_visual = 0.5 * visual_logits_aux + 0.5 * boost_visual
    return specialist_audio, specialist_visual, boost_audio, boost_visual


def resolve_single_run_repair_weights(args, conf_audio, conf_visual):
    if args.sr_target == 'audio':
        return {
            'weight_audio': float(args.sr_audio_scale),
            'weight_visual': 0.0,
            'gap_audio': 0.0,
            'gap_visual': 0.0,
        }
    if args.sr_target == 'visual':
        return {
            'weight_audio': 0.0,
            'weight_visual': float(args.sr_visual_scale),
            'gap_audio': 0.0,
            'gap_visual': 0.0,
        }
    if args.sr_target == 'both':
        return {
            'weight_audio': float(args.sr_audio_scale),
            'weight_visual': float(args.sr_visual_scale),
            'gap_audio': 0.0,
            'gap_visual': 0.0,
        }

    margin = max(0.0, float(args.sr_margin))
    gain = max(0.0, float(args.sr_gain))
    cap = max(1.0, float(args.sr_cap))
    gap_audio = max(0.0, float(conf_visual) - float(conf_audio) - margin)
    gap_visual = max(0.0, float(conf_audio) - float(conf_visual) - margin)

    weight_audio = 0.0
    weight_visual = 0.0
    if gap_audio > 0:
        weight_audio = min(cap, float(args.sr_audio_scale) * (1.0 + gain * gap_audio))
    if gap_visual > 0:
        weight_visual = min(cap, float(args.sr_visual_scale) * (1.0 + gain * gap_visual))

    return {
        'weight_audio': float(weight_audio),
        'weight_visual': float(weight_visual),
        'gap_audio': float(gap_audio),
        'gap_visual': float(gap_visual),
    }
