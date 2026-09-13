"""DeepSeek-V4.1 model support; media/tokenizer workers need no GPU imports."""

from importlib import import_module

_EXPORTS = {
    "DeepseekV41Args": "args", "load_args": "args", "parse_config": "config",
    "DeepseekV41ForCausalLM": "model", "iter_weights": "weight",
    "load_nvfp4_expert_sources": "weight", "load_nvfp4_expert_sources_parallel": "weight",
    "is_expert_tensor": "weight",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
