"""Stage C tests: C0 exact allowlist gadgets & real breast-cancer plausibility, C1 default removal & synthetic_smoke, C2 DATASET_PENDING, C3 UI/telemetry."""

import io
import os
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer

from app.ingestion.exceptions import ConfigurationError, ModelLoadError, UnsafeModelError
from app.ingestion.loader.sklearn_loader import _RestrictedPickleUnpickler
from app.ingestion.metadata.model_metadata import ModelMetadata
from app.ingestion.pipeline import ingest_model
from frontend.services.ingestion_service import (
    _generate_domain_sample_batch,
    format_rejection_guidance,
    load_model_and_dataset,
)


def test_c0_exact_allowlist_blocks_real_gadgets():
    """C0a: Restricted unpickler rejects os.system, builtins.eval, builtins.getattr, and numpy.testing._private.utils.runstring."""
    import numpy.testing._private.utils as nptu

    class _GadgetOs:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    class _GadgetEval:
        def __reduce__(self):
            return (eval, ("1+1",))

    class _GadgetGetattr:
        def __reduce__(self):
            return (getattr, ({}, "keys"))

    class _GadgetNpTest:
        def __reduce__(self):
            return (nptu.runstring, ("x = 1", {}))

    for gadget in (_GadgetOs(), _GadgetEval(), _GadgetGetattr(), _GadgetNpTest()):
        payload = pickle.dumps(gadget)
        with pytest.raises(UnsafeModelError, match="non-allowlisted"):
            _RestrictedPickleUnpickler(io.BytesIO(payload)).load()


def test_c0_real_cancer_prediction_trees_plausibility():
    """C0b: Real scikit-learn/cancer-prediction-trees ingests as ACCEPTED_LIMITED and achieves high accuracy on breast_cancer."""
    res = ingest_model("scikit-learn/cancer-prediction-trees", unsafe_allow_pickle=True)
    assert res.acceptance_level == "ACCEPTED_LIMITED"
    assert getattr(res.adapter, "is_reconstructed", False) is True

    data = load_breast_cancer()
    X, y = data.data, data.target
    probs = res.adapter._model.predict_proba(X)
    preds = res.adapter._model.predict(X)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(preds).tolist()).issubset({0, 1})
    acc = float(np.mean(preds == y))
    assert acc >= 0.90


def test_c1_synthetic_smoke_flag_and_removed_silent_defaults():
    """C1: synthetic_smoke=True only when internal smoke sample is generated; missing dims raise ConfigurationError instead of silent defaults."""
    tiny = nn.Sequential(nn.Linear(4, 3))
    res_with_sample = ingest_model(tiny, sample_input=torch.randn(1, 4), data_domain="tabular")
    assert res_with_sample.metadata.extra_info.get("synthetic_smoke") is False

    res_smoke = ingest_model(tiny, data_domain="tabular")
    assert res_smoke.metadata.extra_info.get("synthetic_smoke") is True

    meta_text = ModelMetadata(model_id="m", model_name="m", data_domain="text", framework="huggingface")
    with pytest.raises(ConfigurationError, match="seq_len="):
        _generate_domain_sample_batch(meta_text, sample_size=2, raw_model=None)

    meta_img = ModelMetadata(model_id="m", model_name="m", data_domain="image", framework="pytorch")
    with pytest.raises(ConfigurationError, match="input_shape="):
        _generate_domain_sample_batch(meta_img, sample_size=2, raw_model=None)


def test_c2_dataset_failure_leaves_model_accepted_dataset_pending(tmp_path: Path):
    """C2: Dataset resolution failure never fails model acceptance; returns DATASET_PENDING and no 'custom_tabular' placeholder."""
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, 2, 20)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    p = tmp_path / "clf.joblib"
    joblib.dump(clf, str(p))

    res = load_model_and_dataset(
        model_identifier=str(p),
        dataset_name="nonexistent/missing_dataset_xyz",
        data_domain="tabular",
        framework="sklearn",
        unsafe_allow_pickle=True,
        allow_synthetic_fallback=False,
    )
    assert res["adapter"] is not None
    assert res["acceptance_level"] == "ACCEPTED_LIMITED"
    assert res["dataset_status"] == "DATASET_PENDING"
    assert res["dataset_resolved"] is False
    assert "custom_tabular" not in str(res.get("dataset_name", ""))


def test_c3_rejection_guidance_and_streamlit_ui_elements():
    """C3: R1..R6 guidance returns summary + next_step; removed UI warnings/expanders stay absent."""
    for code in ("R1", "R2", "R3", "R4", "R5", "R6"):
        err = ModelLoadError("err", rejection_reason=code)
        g = format_rejection_guidance(err)
        assert g["reason"] == code
        assert g["summary"] and g["next_step"]

    src = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")
    assert "acceptance_level" in src
    assert "_requires_remote_code_consent_for" in src
    assert "Retry ingestion" in src
    assert "Add an evaluation dataset to run attacks." in src
    for removed in (
        "Model-Only Ingestion Mode",
        "0 REAL SAMPLES",
        "Evaluation Dataset Required",
        "CRITICAL WARNING — SYNTHETIC FALLBACK DATA IN USE",
        "Explicitly generate synthetic smoke-test data",
        'st.expander("Technical details")',
    ):
        assert removed not in src


def test_c3_build_ingestion_kwargs_and_signature_alignment():
    """Verify build_ingestion_kwargs keys match load_model_and_dataset, ingest_model, and HuggingFaceLoader.load signatures."""
    import inspect
    from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
    from app.ingestion.loader.huggingface_loader import HuggingFaceLoader
    from app.ingestion.pipeline import IngestionResult, ingest_model
    from frontend.streamlit_app import build_ingestion_kwargs

    kwargs = build_ingestion_kwargs(
        model_identifier="org/tiny-hf",
        dataset_input="",
        data_domain="image",
        framework="huggingface",
        device="cpu",
        sample_size=4,
    )
    assert kwargs["trust_remote_code"] is False

    sig_service = inspect.signature(load_model_and_dataset)
    sig_pipeline = inspect.signature(ingest_model)
    sig_hf = inspect.signature(HuggingFaceLoader.load)

    for k in kwargs:
        assert k in sig_service.parameters, f"UI kwarg '{k}' missing from load_model_and_dataset"

    for fn_name, sig in (("load_model_and_dataset", sig_service), ("ingest_model", sig_pipeline), ("HuggingFaceLoader.load", sig_hf)):
        assert "trust_remote_code" in sig.parameters, f"trust_remote_code missing from {fn_name}"
        assert sig.parameters["trust_remote_code"].default is False

    tiny = nn.Sequential(nn.Linear(4, 2))
    mock_meta = ModelMetadata(model_id="org/tiny-hf", model_name="tiny-hf", data_domain="tabular", framework="huggingface", num_classes=2)
    mock_res = IngestionResult(adapter=PyTorchAdapter(tiny), metadata=mock_meta)

    with patch("frontend.services.ingestion_service.ingest_model", return_value=mock_res) as mock_ingest:
        res = load_model_and_dataset(**kwargs)
    assert res["adapter"] is not None
    assert mock_ingest.call_args.kwargs["trust_remote_code"] is False


def test_c3_remote_code_consent_only_when_required():
    """Verify requires_remote_code is True only when auto_map is present in HF config."""
    from frontend.services.ingestion_service import inspect_model_summary

    with patch("app.ingestion.utils.model_utils.detect_hf_repo_info", return_value={
        "framework": "huggingface", "domain": "image", "files": ["config.json"], "is_model_repo": True, "requires_remote_code": True
    }):
        assert inspect_model_summary("org/custom-automap")["requires_remote_code"] is True

    with patch("app.ingestion.utils.model_utils.detect_hf_repo_info", return_value={
        "framework": "sklearn", "domain": "tabular", "files": ["model.joblib"], "is_model_repo": True, "requires_remote_code": False
    }):
        assert inspect_model_summary("org/sklearn-repo")["requires_remote_code"] is False


def test_c3_streamlit_apptest_hf_ingestion_no_exception():
    """Drive the ingestion page via Streamlit AppTest with HF source and mocked ingest_model."""
    from streamlit.testing.v1 import AppTest
    from app.ingestion.adapters.pytorch_adapter import PyTorchAdapter
    from app.ingestion.pipeline import IngestionResult

    tiny = nn.Sequential(nn.Linear(4, 2))
    mock_meta = ModelMetadata(model_id="org/tiny-hf", model_name="tiny-hf", data_domain="tabular", framework="huggingface", num_classes=2)
    mock_res = IngestionResult(adapter=PyTorchAdapter(tiny), metadata=mock_meta, acceptance_level="ACCEPTED_FULL")

    with (
        patch("app.ingestion.pipeline.ingest_model", return_value=mock_res),
        patch("frontend.services.ingestion_service.ingest_model", return_value=mock_res),
        patch("app.ingestion.utils.model_utils.detect_hf_repo_info", return_value={
            "framework": "huggingface", "domain": "tabular", "files": ["config.json"], "is_model_repo": True, "requires_remote_code": False, "error": None, "rejection_reason": None
        }),
    ):
        app_path = str(Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py")
        at = AppTest.from_file(app_path, default_timeout=20)
        at.run()
        # Navigate to Stage 1: Model Ingestion
        start_btns = [b for b in at.button if "Model Ingestion" in b.label or "Start Model Assessment" in b.label]
        assert start_btns
        start_btns[0].click()
        at.run()

        # Select Hugging Face Model ID radio if present
        hf_radios = [r for r in at.radio if "Hugging Face Model ID" in (r.options or [])]
        if hf_radios:
            hf_radios[0].set_value("Hugging Face Model ID")
            at.run()

        # Enter HF repo ID
        hf_inputs = [t for t in at.text_input if "Hugging Face" in t.label]
        if hf_inputs:
            hf_inputs[0].set_value("org/tiny-hf")
            at.run()

        # Click Ingest & Validate Model
        ingest_btns = [b for b in at.button if "Ingest & Validate Model" in b.label]
        assert ingest_btns
        ingest_btns[0].click()
        at.run()
        assert len(at.exception) == 0

        # Verify removed strings are absent and synthetic is never used
        all_text = " ".join(
            [getattr(e, "value", "") for e in (list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))]
        )
        for removed in (
            "Model-Only Ingestion Mode",
            "0 REAL SAMPLES",
            "Evaluation Dataset Required",
            "CRITICAL WARNING",
            "Explicitly generate synthetic",
            "Technical details",
        ):
            assert removed not in all_text

    # Also verify sample count shown when resolved (mocked) and nothing shown when unresolved
    with (
        patch("frontend.services.ingestion_service.get_dataset_population_info", return_value={
            "total_samples": 120, "resolved": True, "source_type": "mock", "resolved_source": "org/ds", "provenance": "mock"
        }),
        patch("app.ingestion.utils.model_utils.detect_hf_repo_info", return_value={
            "framework": "huggingface", "domain": "tabular", "files": ["config.json"], "is_model_repo": True, "requires_remote_code": False, "error": None, "rejection_reason": None
        }),
    ):
        at_res = AppTest.from_file(app_path, default_timeout=20)
        at_res.run()
        [b for b in at_res.button if "Model Ingestion" in b.label or "Start Model Assessment" in b.label][0].click()
        at_res.run()
        hf_radios = [r for r in at_res.radio if "Hugging Face Model ID" in (r.options or [])]
        if hf_radios:
            hf_radios[0].set_value("Hugging Face Model ID")
            at_res.run()
        assert any("120" in getattr(m, "value", "") for m in at_res.markdown)

    with (
        patch("frontend.services.ingestion_service.get_dataset_population_info", return_value={
            "total_samples": 0, "resolved": False, "source_type": "none", "resolved_source": None, "provenance": "none"
        }),
        patch("app.ingestion.utils.model_utils.detect_hf_repo_info", return_value={
            "framework": "huggingface", "domain": "tabular", "files": ["config.json"], "is_model_repo": True, "requires_remote_code": False, "error": None, "rejection_reason": None
        }),
    ):
        at_unres = AppTest.from_file(app_path, default_timeout=20)
        at_unres.run()
        [b for b in at_unres.button if "Model Ingestion" in b.label or "Start Model Assessment" in b.label][0].click()
        at_unres.run()
        hf_radios = [r for r in at_unres.radio if "Hugging Face Model ID" in (r.options or [])]
        if hf_radios:
            hf_radios[0].set_value("Hugging Face Model ID")
            at_unres.run()
        unres_text = " ".join(getattr(e, "value", "") for e in (list(at_unres.info) + list(at_unres.warning) + list(at_unres.markdown)))
        assert "0 REAL SAMPLES" not in unres_text
        assert "Model-Only Ingestion Mode" not in unres_text

