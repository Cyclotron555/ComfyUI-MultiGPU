"""Prepare pre-encoded controls matching core WanVaceToVideo (no model loading).

This optional worker utility reuses an existing FP32 Wan VAE in a separate
process. It does not add pose estimation or guarantee motion quality.
"""
import math
import torch
import node_helpers
from comfy.latent_formats import Wan21


def prepare_conditioning(positive, negative, inactive, reactive, *, frames, width,
                         height, strength=1.0, reference=None):
    if (frames < 1 or width % 8 or height % 8 or
            min(width, height) < 8 or not math.isfinite(strength)):
        raise ValueError('Invalid VACE frame geometry or control strength')
    shape = (1, 16, (frames - 1) // 4 + 1, height // 8, width // 8)
    for value in (inactive, reactive):
        if tuple(value.shape) != shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError('Invalid encoded VACE control tensor')
        if value.device.type != 'cpu':
            raise ValueError('Pre-encoded worker controls must be on CPU')
    video = torch.cat((inactive, reactive), dim=1)
    mask = torch.ones((1, 64, shape[2], shape[3], shape[4]))
    trim = 0
    if reference is not None:
        if (tuple(reference.shape) != (1, 16, 1, shape[3], shape[4]) or
                reference.device.type != 'cpu' or not reference.is_floating_point() or
                not torch.isfinite(reference).all()):
            raise ValueError('Invalid single-frame reference latent')
        neutral = Wan21().process_out(torch.zeros_like(reference))
        video = torch.cat((torch.cat((reference, neutral), dim=1), video), dim=2)
        mask = torch.cat((torch.zeros_like(mask[:, :, :1]), mask), dim=2)
        trim = 1
    values = {'vace_frames': [video], 'vace_mask': [mask], 'vace_strength': [strength]}
    positive = node_helpers.conditioning_set_values(positive, values, append=True)
    negative = node_helpers.conditioning_set_values(negative, values, append=True)
    latent = {'samples': torch.zeros((1, 16, shape[2] + trim, shape[3], shape[4]))}
    return positive, negative, latent, trim
