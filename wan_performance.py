"""Opt-in Wan performance settings using native ComfyUI model-patcher APIs.

No downloads, global attention replacement, dtype mutation or device selection.
The existing safe launcher owns UUID mapping and precision selection.
"""
from importlib import import_module
import math


def launch_flags(enabled=False):
    """BF16 for an explicitly selected Wan session; baseline stays unchanged."""
    return ['--bf16-unet', '--use-pytorch-cross-attention', '--disable-dynamic-vram'] if enabled else []


def configure_wan(model, *, attention='pytorch', cache_threshold=0.0,
                  cache_start=0.15, cache_end=0.85):
    """Return a cloned native model patcher and an explicit configuration report.

    INT8 attention and EasyCache alter numerical results. Both are opt-in.
    Unsupported optional components fall back with a reason in the report;
    model-patching errors propagate instead of silently changing the workload.
    """
    if attention not in ('pytorch', 'kitchen_int8'):
        raise ValueError('Choose pytorch or kitchen_int8 attention')
    if not all(math.isfinite(value) for value in (cache_threshold, cache_start, cache_end)):
        raise ValueError('Cache settings must be finite')
    if cache_threshold < 0 or not 0 <= cache_start < cache_end <= 1:
        raise ValueError('Cache threshold must be nonnegative and its window must satisfy 0 <= start < end <= 1')
    diffusion = getattr(getattr(model, 'model', None), 'diffusion_model', None)
    if diffusion is None or not type(diffusion).__module__.startswith('comfy.ldm.wan.'):
        raise ValueError('This tested performance helper is for native ComfyUI Wan models')
    report = {'requested_attention': attention, 'attention': 'pytorch',
              'cache_threshold': 0.0, 'cache_window': [cache_start, cache_end],
              'compute_device': str(model.load_device), 'fallback_reasons': [],
              'note': 'Opt-in approximations require a matched visual test; no universal speed or quality guarantee.'}
    native_attention = import_module('comfy.ldm.modules.attention')
    function = native_attention.get_attention_function('pytorch')
    if attention == 'kitchen_int8':
        candidate = native_attention.get_attention_function('comfy_kitchen_int8', None)
        try:
            kitchen = import_module('comfy_kitchen')
            supported = kitchen.int8_attention_is_available(model.load_device)
        except (ImportError, AttributeError):
            supported = False
        if candidate is not None and supported:
            function = candidate
            report['attention'] = 'kitchen_int8'
        else:
            report['fallback_reasons'].append('Native INT8 attention unavailable on the model compute device; using PyTorch')
    patched = model.clone()
    patched.set_model_optimized_attention(function)
    if cache_threshold:
        try:
            cache_node = import_module('comfy_extras.nodes_easycache').EasyCacheNode
        except (ImportError, AttributeError):
            report['fallback_reasons'].append('Native EasyCache unavailable; cache disabled')
        else:
            patched = cache_node.execute(patched, reuse_threshold=cache_threshold,
                                         start_percent=cache_start, end_percent=cache_end,
                                         verbose=False)[0]
            report['cache_threshold'] = cache_threshold
    return patched, report
