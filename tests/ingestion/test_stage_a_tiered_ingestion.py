"""Stage A tests for tiered loading, A1 security, A2 R1 classification, A3 IngestionResult, A4 ranked fallback, A5 exception chains, and A6 error classification."""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from app.ingestion.artifact_detector import ModelArtifactDetector, RankedCandidate
from app.ingestion.exceptions import (
    ErrorCategory,
    MissingDependencyError,
    ModelLoadError,
    UnsafeModelError,
    classify_exception,
)
from app.ingestion.loader.huggingface_loader import HuggingFaceLoader
from app.ingestion.loader.tree_loader import TreeModelLoader
from app.ingestion.pipeline import IngestionResult, ingest_model
from app.ingestion.utils.model_utils import clear_hf_repo_cache, detect_hf_repo_info


def test_a1_no_trust_remote_code_without_consent():
    """A1: Verify no fallback passes trust_remote_code=True unless explicitly consented."""
    seen_trust_flags = []

    def fake_from_pretrained(*args, **kwargs):
        seen_trust_flags.append(kwargs.get("trust_remote_code", False))
        raise RuntimeError("Loading requires you to execute the configuration file in that repo with trust_remote_code=True.")

    fake_hf_info = {
        "framework": "huggingface",
        "domain": "image",
        "task_type": "classification",
        "files": ["config.json", "model.safetensors"],
        "is_model_repo": True,
        "repo_type": "model",
        "error": None,
        "rejection_reason": None,
    }

    with (
        patch("app.ingestion.utils.model_utils.detect_hf_repo_info", return_value=fake_hf_info),
        patch("app.ingestion.loader.huggingface_loader.AutoModelForImageClassification.from_pretrained", side_effect=fake_from_pretrained),
        patch("app.ingestion.loader.huggingface_loader.AutoModel.from_pretrained", side_effect=fake_from_pretrained),
    ):
        with pytest.raises(ModelLoadError) as exc_info:
            HuggingFaceLoader().load("org/custom-code-model", domain="image")

    assert all(flag is False for flag in seen_trust_flags)
    assert exc_info.value.rejection_reason == "R3"
    assert exc_info.value.category == ErrorCategory.CONSENT


def test_a2_401_retry_without_token_and_r1_classification():
    """A2: 401 retries once without token; 404 and gated repos map to R1 and never fall through."""
    clear_hf_repo_cache()

    class FakeResp:
        def __init__(self, code):
            self.status_code = code

    class FakeHTTP401(Exception):
        def __init__(self):
            super().__init__("401 Client Error: Unauthorized")
            self.response = FakeResp(401)

    class FakeHTTP403(Exception):
        def __init__(self):
            super().__init__("403 Client Error: Gated repo")
            self.response = FakeResp(403)

    class FakeHTTP404(Exception):
        def __init__(self):
            super().__init__("404 Client Error: Repository Not Found")
            self.response = FakeResp(404)

    # 1. 401 on first call, succeeds on token=False retry
    mock_api_auth = MagicMock()
    mock_api_auth.list_repo_files.side_effect = FakeHTTP401()
    mock_api_anon = MagicMock()
    mock_api_anon.list_repo_files.return_value = ["README.md", "config.json", "model.safetensors"]
    mock_api_anon.model_info.return_value = MagicMock(library_name="transformers", pipeline_tag="image-classification", tags=[], card_data={})

    with patch("huggingface_hub.HfApi", side_effect=[mock_api_auth, mock_api_anon]):
        info = detect_hf_repo_info("org/public-repo-stale-token", force_refresh=True)
    assert info["rejection_reason"] is None
    assert info["is_model_repo"] is True
    mock_api_anon.list_repo_files.assert_called_once_with("org/public-repo-stale-token", token=False)

    # 2. 404 -> R1
    mock_api_404 = MagicMock()
    mock_api_404.list_repo_files.side_effect = FakeHTTP404()
    with patch("huggingface_hub.HfApi", return_value=mock_api_404):
        info_404 = detect_hf_repo_info("org/missing-repo", force_refresh=True)
    assert info_404["repo_type"] == "not_found"
    assert info_404["rejection_reason"] == "R1"

    with (
        patch("app.ingestion.utils.model_utils.detect_hf_repo_info", return_value=info_404),
        patch("app.ingestion.loader.huggingface_loader.AutoModel.from_pretrained") as mock_auto,
    ):
        with pytest.raises(ModelLoadError) as exc_404:
            HuggingFaceLoader().load("org/missing-repo", domain="image")
    assert exc_404.value.rejection_reason == "R1"
    mock_auto.assert_not_called()

    # 3. Gated 403 -> R1
    mock_api_403 = MagicMock()
    mock_api_403.list_repo_files.side_effect = FakeHTTP403()
    with patch("huggingface_hub.HfApi", return_value=mock_api_403):
        info_403 = detect_hf_repo_info("org/gated-repo", force_refresh=True)
    assert info_403["repo_type"] == "private_or_gated"
    assert info_403["rejection_reason"] == "R1"


def test_a3_ingestion_result_fields_and_tuple_unpacking():
    """A3: IngestionResult exposes acceptance_level, rejection_reason, attempts, warnings while unpacking."""
    tiny = nn.Sequential(nn.Linear(4, 2))
    res = ingest_model(tiny, sample_input=torch.randn(1, 4), data_domain="tabular")
    assert isinstance(res, IngestionResult)
    adapter, metadata = res
    assert adapter is res.adapter
    assert metadata is res.metadata
    assert res.acceptance_level == "ACCEPTED_FULL"
    assert res.rejection_reason is None
    assert isinstance(res.attempts, list)
    assert isinstance(res.warnings, list)


def test_a4_ranked_candidates_autotrain_and_fallback_attempt_log(tmp_path: Path):
    """A4: Ranked candidates prioritize joblib/sklearn over HF; fallback succeeds and logs failed attempt; all-fail returns full attempt log."""
    # 1. AutoTrain config.json + model.joblib -> sklearn/tree ranked ahead of huggingface
    jb_path = tmp_path / "model.joblib"
    with open(jb_path, "wb") as f:
        pickle.dump({"a": 1}, f)
    cands = ModelArtifactDetector.rank_candidates(
        files=["config.json", "model.joblib"],
        local_dir=tmp_path,
        config_dict={"features": ["age", "income"]},
    )
    assert cands[0].loader in ("sklearn", "tree")

    # 2. Wrong loader first fails, second loader succeeds -> attempts logged
    class DummySklearn:
        classes_ = [0, 1]
        n_features_in_ = 4
        def predict(self, x):
            return [0] * len(x)
        def predict_proba(self, x):
            return [[0.8, 0.2]] * len(x)

    fake_hf_info = {
        "framework": "huggingface",
        "domain": "tabular",
        "task_type": "classification",
        "files": ["model.joblib", "config.json"],
        "is_model_repo": True,
        "repo_type": "model",
        "error": None,
        "rejection_reason": None,
    }
    ranked = [
        RankedCandidate("tree", "model.joblib", 0.95, ["mock tree first"], "xgboost"),
        RankedCandidate("sklearn", "model.joblib", 0.90, ["mock sklearn second"], "sklearn"),
    ]
    with (
        patch("app.ingestion.utils.model_utils.detect_hf_repo_info", return_value=fake_hf_info),
        patch("app.ingestion.utils.model_utils.detect_model_framework", return_value="huggingface"),
        patch("app.ingestion.utils.model_utils.detect_model_domain", return_value="tabular"),
        patch("app.ingestion.artifact_detector.ModelArtifactDetector.rank_candidates", return_value=ranked),
        patch("app.ingestion.loader.tree_loader.TreeModelLoader.load", side_effect=ModelLoadError("not an xgboost json")),
        patch("app.ingestion.loader.sklearn_loader.SklearnLoader.load", return_value=DummySklearn()),
    ):
        res = ingest_model("org/fake-autotrain-repo", data_domain="tabular", framework="huggingface", unsafe_allow_pickle=True)
    assert res.acceptance_level == "ACCEPTED_LIMITED"
    assert len(res.attempts) == 1
    assert res.attempts[0]["loader"] == "tree"

    # 3. All candidates fail -> ModelLoadError has full attempt log and preserved __cause__
    with (
        patch("app.ingestion.utils.model_utils.detect_hf_repo_info", return_value=fake_hf_info),
        patch("app.ingestion.utils.model_utils.detect_model_framework", return_value="huggingface"),
        patch("app.ingestion.utils.model_utils.detect_model_domain", return_value="tabular"),
        patch("app.ingestion.artifact_detector.ModelArtifactDetector.rank_candidates", return_value=ranked),
        patch("app.ingestion.loader.tree_loader.TreeModelLoader.load", side_effect=ModelLoadError("tree fail")),
        patch("app.ingestion.loader.sklearn_loader.SklearnLoader.load", side_effect=ValueError("bad pickle")),
    ):
        with pytest.raises(ModelLoadError) as all_fail:
            ingest_model("org/fake-broken-repo", data_domain="tabular", framework="huggingface", unsafe_allow_pickle=True)
    assert len(all_fail.value.attempts) >= 2
    assert all_fail.value.__cause__ is not None


def test_a5_exception_chain_preserved(tmp_path: Path):
    """A5: ImportError -> ModelLoadError preserves __cause__."""
    dummy_xgb = tmp_path / "model.json"
    dummy_xgb.write_text('{"learner": {}}', encoding="utf-8")
    with patch.dict("sys.modules", {"xgboost": None}):
        with pytest.raises(ModelLoadError) as exc_info:
            TreeModelLoader().load(dummy_xgb, framework="xgboost")
    assert isinstance(exc_info.value.__cause__, ImportError)


def test_a6_error_category_classification():
    """A6: classify_exception maps error types and messages to ErrorCategory."""
    assert classify_exception(ValueError("node array from the pickle has an incompatible dtype")) == ErrorCategory.VERSION_MISMATCH
    assert classify_exception(ImportError("No module named 'onnxruntime'")) == ErrorCategory.MISSING_DEPENDENCY
    assert classify_exception(UnsafeModelError("Refusing without explicit user consent")) == ErrorCategory.CONSENT
    assert classify_exception(RuntimeError("state_dict missing architecture")) == ErrorCategory.ARCH_UNKNOWN
    assert classify_exception(ValueError("Validation error for field 'id2label'")) == ErrorCategory.METADATA_INVALID
    assert classify_exception(ValueError("not a zip file / corrupt header")) == ErrorCategory.MALFORMED
    assert classify_exception(MissingDependencyError("install xgboost")) == ErrorCategory.MISSING_DEPENDENCY
    assert classify_exception(RuntimeError("unexpected failure")) == ErrorCategory.OTHER

