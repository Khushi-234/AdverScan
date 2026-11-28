"""
Regression tests for Hugging Face Joblib security consent propagation
and frontend session/cache state reset on model switches and retries.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from app.ingestion.exceptions import UnsafeModelError
from app.ingestion.utils.model_utils import (
    _HF_REPO_INFO_CACHE,
    clear_hf_repo_cache,
    detect_hf_repo_info,
)
from frontend.services.ingestion_service import (
    inspect_model_summary,
    load_model_and_dataset,
)
from frontend.utils.session_state import (
    clear_active_assessment,
    get_active_assessment,
    set_active_assessment,
)


def test_joblib_security_consent_blocked_by_default_and_allowed_with_consent(tmp_path: Path):
    """Issue 1: unsafe_allow_pickle defaults to False (blocked) and reaches SklearnLoader when True."""
    model_path = tmp_path / "model.joblib"
    clf = LogisticRegression().fit(np.random.randn(20, 4), np.random.randint(0, 2, size=20))
    joblib.dump(clf, model_path)

    # 1. Default (without explicit consent) must raise UnsafeModelError
    with pytest.raises(UnsafeModelError, match="without explicit user consent"):
        load_model_and_dataset(
            model_identifier=str(model_path),
            data_domain="tabular",
            framework="sklearn",
            sample_size=4,
        )

    # 2. Explicit user consent passes unsafe_allow_pickle=True through pipeline to SklearnLoader
    res = load_model_and_dataset(
        model_identifier=str(model_path),
        data_domain="tabular",
        framework="sklearn",
        sample_size=4,
        unsafe_allow_pickle=True,
        allow_synthetic_fallback=True,
    )
    assert res["model_identifier"] == str(model_path)
    assert res["adapter"] is not None
    assert res["test_inputs"].shape == (4, 4)


def test_inspect_model_summary_requires_pickle_consent_only_for_pickle_formats(tmp_path: Path):
    """Issue 1: Consent is required for joblib/pkl, NOT for safe formats like XGBoost JSON or ONNX."""
    joblib_file = tmp_path / "safe_check.joblib"
    joblib_file.touch()
    local_summary = inspect_model_summary(str(joblib_file))
    assert local_summary["requires_pickle_consent"] is True

    # Mock HF repo with joblib -> requires consent
    with patch(
        "app.ingestion.utils.model_utils.detect_hf_repo_info",
        return_value={
            "framework": "sklearn",
            "domain": "tabular",
            "model_file": "model.joblib",
            "weight_files": ["model.joblib"],
            "is_model_repo": True,
            "files": ["README.md", "model.joblib"],
        },
    ):
        hf_sklearn = inspect_model_summary("some-org/sklearn-joblib-model")
        assert hf_sklearn["requires_pickle_consent"] is True

    # Mock HF repo with native XGBoost JSON -> NO consent required
    with patch(
        "app.ingestion.utils.model_utils.detect_hf_repo_info",
        return_value={
            "framework": "xgboost",
            "domain": "tabular",
            "model_file": "xgb_model.json",
            "weight_files": ["xgb_model.json"],
            "is_model_repo": True,
            "files": ["README.md", "xgb_model.json"],
        },
    ):
        hf_xgb = inspect_model_summary("HarryDataScience/sentinel-fraud-xgboost")
        assert hf_xgb["requires_pickle_consent"] is False

    # Mock HF repo with ONNX -> NO consent required
    with patch(
        "app.ingestion.utils.model_utils.detect_hf_repo_info",
        return_value={
            "framework": "onnx",
            "domain": "image",
            "model_file": "model.onnx",
            "weight_files": ["model.onnx"],
            "is_model_repo": True,
            "files": ["README.md", "model.onnx"],
        },
    ):
        hf_onnx = inspect_model_summary("some-org/resnet-onnx")
        assert hf_onnx["requires_pickle_consent"] is False


def test_hf_repo_cache_not_poisoned_by_errors_and_force_refresh_clears_cache():
    """Issue 2: Failed HF lookups are not cached and force_refresh clears stale entries."""
    repo_id = "org/transient-hf-model"
    clear_hf_repo_cache(repo_id)

    with patch("huggingface_hub.HfApi.list_repo_files", side_effect=RuntimeError("network error")):
        info_err = detect_hf_repo_info(repo_id)
        assert info_err["error"] is not None
        assert repo_id not in _HF_REPO_INFO_CACHE

    mock_info = MagicMock()
    mock_info.library_name = "onnx"
    mock_info.pipeline_tag = "image-classification"
    mock_info.tags = ["onnx"]

    with patch("huggingface_hub.HfApi.list_repo_files", return_value=["README.md", "model.onnx"]):
        with patch("huggingface_hub.HfApi.model_info", return_value=mock_info):
            info_ok = detect_hf_repo_info(repo_id, force_refresh=True)
            assert info_ok["framework"] == "onnx"
            assert repo_id in _HF_REPO_INFO_CACHE

    clear_hf_repo_cache(repo_id)
    assert repo_id not in _HF_REPO_INFO_CACHE


def test_session_state_clear_active_assessment_resets_stale_model_state():
    """Issue 2: clear_active_assessment clears previous model metadata and downstream results."""
    import streamlit as st

    st.session_state.clear()
    st.session_state["theme"] = "Light"

    set_active_assessment(
        {
            "model_identifier": "org/model-a",
            "adapter": MagicMock(),
            "metadata": MagicMock(),
            "metadata_dict": {"Model Name": "model-a", "Framework": "PYTORCH"},
            "domain": "image",
        }
    )
    assert get_active_assessment() is not None
    assert st.session_state["active_model_name"] == "model-a"

    clear_active_assessment()
    assert get_active_assessment() is None
    assert st.session_state["is_assessment_active"] is False
    assert st.session_state["active_model_name"] is None
    assert st.session_state["active_model_domain"] is None
    assert st.session_state["theme"] == "Light"

