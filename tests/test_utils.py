"""Model config loading: the subset of the HF config.json the simulator needs.
Was the ``__main__`` block of ``serving/core/utils.py``."""

from llmservingsim.serving.core.utils import get_config


def test_get_config():
    config = get_config("meta-llama/Llama-3.1-8B")
    assert config["model_type"] == "llama", config.get("model_type")
    for key in ("hidden_size", "num_attention_heads", "num_hidden_layers",
                "num_key_value_heads", "intermediate_size", "vocab_size"):
        assert isinstance(config[key], int), (key, config.get(key))
