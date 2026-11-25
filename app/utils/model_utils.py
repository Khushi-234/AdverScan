"""
Model utility functions for patching Hugging Face configurations and standardizing model loading across AdverScan.
"""

import json
from typing import Tuple, Dict, Any
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageClassification, ViTConfig


def patch_hf_config(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Patch Hugging Face model configuration dictionary.
    Specifically handles null value in id2label (e.g. GTSRB ViT model class 43 mapping issue)
    and re-calculates label2id and num_labels.

    Args:
        cfg_dict: Raw configuration dictionary from model's config.json.

    Returns:
        Patched configuration dictionary.
    """
    if "id2label" in cfg_dict and isinstance(cfg_dict["id2label"], dict):
        patched_id2label = {}
        for k, v in cfg_dict["id2label"].items():
            try:
                int_key = int(k)
            except (TypeError, ValueError):
                int_key = k
            label_val = str(v).strip() if v is not None else ""
            if not label_val or label_val.lower() == "none":
                label_val = f"Class {int_key}"
            if int_key in patched_id2label and patched_id2label[int_key] != label_val:
                raise ValueError(
                    f"Conflicting duplicate id2label key after integer conversion: "
                    f"{int_key} ('{patched_id2label[int_key]}' vs '{label_val}')"
                )
            patched_id2label[int_key] = label_val

        cfg_dict["id2label"] = patched_id2label
        cfg_dict["label2id"] = {
            str(v): k for k, v in patched_id2label.items() if isinstance(k, int)
        }
        cfg_dict["num_labels"] = len(patched_id2label)

    return cfg_dict


def load_gtsrb_vit_model(model_name: str = "bazyl/gtsrb-model") -> Tuple[AutoModelForImageClassification, ViTConfig]:
    """
    Fetch config, apply upstream patches, and load pre-trained Vision Transformer model.

    Args:
        model_name: Hugging Face model hub identifier.

    Returns:
        Tuple of (AutoModelForImageClassification, ViTConfig).
    """
    config_path = hf_hub_download(model_name, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    cfg_dict = patch_hf_config(cfg_dict)
    model_config = ViTConfig.from_dict(cfg_dict)
    raw_model = AutoModelForImageClassification.from_pretrained(
        model_name, config=model_config, use_safetensors=True
    )
    return raw_model, model_config
