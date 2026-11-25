"""Regression tests for Hugging Face config normalization (e.g., bazyl/gtsrb-model)."""

from pathlib import Path
from PIL import Image
import pytest
import torch

from transformers import CONFIG_MAPPING
from app.ingestion.loader.huggingface_loader import patch_hf_config
from app.ingestion.pipeline import ingest_model
from frontend.services.ingestion_service import load_model_and_dataset


GTSRB_RAW_ID2LABEL = {
    "0": "Speed limit (20km/h)",
    "1": "Speed limit (30km/h)",
    "10": "No passing veh over 3.5 tons",
    "11": "Right-of-way at intersection",
    "12": "Priority road",
    "13": "Yield",
    "14": "Stop",
    "15": "No vehicles",
    "16": "Veh > 3.5 tons prohibited",
    "17": "No entry",
    "18": "General caution",
    "19": "Dangerous curve left",
    "2": "Speed limit (50km/h)",
    "20": "Dangerous curve right",
    "21": "Double curve",
    "22": "Bumpy road",
    "23": "Slippery road",
    "24": "Road narrows on the right",
    "25": "Road work",
    "26": "Traffic signals",
    "27": "Pedestrians",
    "28": "Children crossing",
    "29": "Bicycles crossing",
    "3": "Speed limit (60km/h)",
    "30": "Beware of ice/snow",
    "31": "Wild animals crossing",
    "32": "End speed + passing limits",
    "33": "Turn right ahead",
    "34": "Turn left ahead",
    "35": "Ahead only",
    "36": "Go straight or right",
    "37": "Go straight or left",
    "38": "Keep right",
    "39": "Keep left",
    "4": "Speed limit (70km/h)",
    "40": "Roundabout mandatory",
    "41": "End of no passing",
    "42": "End no passing veh > 3.5 tons",
    "43": None,
    "5": "Speed limit (80km/h)",
    "6": "End of speed limit (80km/h)",
    "7": "Speed limit (100km/h)",
    "8": "Speed limit (120km/h)",
    "9": "No passing",
}


def test_patch_hf_config_gtsrb_exact_metadata_and_generic_models():
    """Verify id2label normalization converts keys to int, fills None with Class 43, and validates generically."""
    for model_type in ("vit", "resnet", "convnext"):
        raw_cfg = {
            "model_type": model_type,
            "id2label": dict(GTSRB_RAW_ID2LABEL),
        }
        patched = patch_hf_config(raw_cfg)

        assert len(patched["id2label"]) == 44
        assert patched["num_labels"] == 44
        assert all(isinstance(k, int) for k in patched["id2label"].keys())
        assert patched["id2label"][0] == "Speed limit (20km/h)"
        assert patched["id2label"][43] == "Class 43"
        assert patched["label2id"]["Class 43"] == 43
        assert patched["label2id"]["Speed limit (20km/h)"] == 0

        cfg_obj = CONFIG_MAPPING[model_type].from_dict(patched)
        assert cfg_obj.num_labels == 44
        assert cfg_obj.id2label[43] == "Class 43"
        assert cfg_obj.label2id["Class 43"] == 43


def test_patch_hf_config_rejects_conflicting_duplicate_keys():
    """Verify genuine conflicts after integer key conversion raise a clear ValueError."""
    raw_cfg = {
        "model_type": "vit",
        "id2label": {
            "0": "Stop",
            0: "Yield",
        },
    }
    with pytest.raises(ValueError, match="Conflicting duplicate id2label key after integer conversion: 0"):
        patch_hf_config(raw_cfg)


def test_bazyl_gtsrb_model_ingestion_and_image_inference():
    """Verify bazyl/gtsrb-model ingests end-to-end (including via load_model_and_dataset) and runs inference."""
    res = load_model_and_dataset(
        model_identifier="bazyl/gtsrb-model",
        dataset_name="bazyl/GTSRB",
        data_domain="image",
        framework="huggingface",
        sample_size=5,
    )
    adapter = res["adapter"]
    metadata = res["metadata"]
    assert metadata.framework == "huggingface"
    assert metadata.data_domain == "image"
    assert metadata.num_classes == 44
    cfg = adapter.get_model().config
    assert len(cfg.id2label) == 44
    assert all(isinstance(k, int) for k in cfg.id2label.keys())
    assert cfg.id2label[43] == "Class 43"

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    out = adapter.predict(img)
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (1, 44)


def test_frontend_removed_upload_evaluation_dataset_section():
    """Verify the 'Or Upload Evaluation Dataset' uploader UI block is removed from streamlit_app.py."""
    app_src = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")
    assert "Or Upload Evaluation Dataset" not in app_src
    assert "dataset_upload::" not in app_src

