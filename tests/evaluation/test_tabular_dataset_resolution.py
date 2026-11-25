"""
Regression tests for tabular dataset resolution, schema/feature ordering,
TabularDatasetLoader(data=...) argument flow, and synthetic fallback evaluation guards.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import torch

from app.evaluation.dataset_loader.factory import (
    _schema_matches_model,
    get_dataset_loader,
    resolve_tabular_dataset_source,
)
from app.evaluation.dataset_loader.tabular_loader import TabularDatasetLoader
from app.ingestion.adapters.tree_adapter import TreeModelAdapter
from app.ingestion.metadata import ModelMetadata
from app.ingestion.pipeline import IngestionResult


def test_hf_repo_bundled_csv_reaches_tabular_loader(tmp_path: Path):
    """Verify HF tabular repo with bundled CSV resolves via hf_hub_download and passes data=... to TabularDatasetLoader."""
    csv_path = tmp_path / "eval_data.csv"
    df = pd.DataFrame(
        {
            "f_b": [1.0, 2.0, 3.0, 4.0],
            "f_a": [0.5, 0.6, 0.7, 0.8],
            "Class": [0, 1, 0, 1],
        }
    )
    df.to_csv(csv_path, index=False)

    mock_hf_info = {
        "framework": "xgboost",
        "domain": "tabular",
        "dataset_files": ["eval_data.csv"],
        "dataset_candidates": [],
        "feature_names": ["f_a", "f_b"],
    }

    with patch("app.ingestion.utils.model_utils.detect_hf_repo_info", return_value=mock_hf_info):
        with patch("huggingface_hub.hf_hub_download", return_value=str(csv_path)) as mock_dl:
            loader = get_dataset_loader(
                dataset_name="custom_dataset",
                data_domain="tabular",
                model_id="org/xgb-with-csv",
                feature_names=["f_a", "f_b"],
                batch_size=4,
            )

    mock_dl.assert_called_once_with("org/xgb-with-csv", "eval_data.csv")
    assert isinstance(loader, TabularDatasetLoader)
    batch_x, batch_y, _ = next(loader.iterate_batches())
    assert batch_x.shape == (4, 2)
    # Verify column reordering: f_a should be col 0 ([0.5, ...]), f_b should be col 1 ([1.0, ...])
    assert torch.allclose(batch_x[:, 0], torch.tensor([0.5, 0.6, 0.7, 0.8]))
    assert torch.allclose(batch_x[:, 1], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert batch_y.tolist() == [0, 1, 0, 1]


def test_hf_metadata_dataset_candidate_dataframe_reaches_tabular_loader():
    """Verify dataset identifier in HF model metadata resolves to DataFrame and passes data=df."""
    mock_hf_info = {
        "framework": "sklearn",
        "domain": "tabular",
        "dataset_files": [],
        "dataset_candidates": ["org/fraud-tabular-ds"],
    }
    df_hf = pd.DataFrame(
        {
            "x1": [10.0, 20.0, 30.0],
            "x2": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        }
    )

    with patch("app.ingestion.utils.model_utils.detect_hf_repo_info", return_value=mock_hf_info):
        with patch(
            "app.evaluation.dataset_loader.factory._load_hf_dataset_as_dataframe",
            return_value=df_hf,
        ) as mock_load_ds:
            loader = get_dataset_loader(
                dataset_name="custom_dataset",
                data_domain="tabular",
                model_id="org/tabular-model",
                feature_names=["x1", "x2"],
                batch_size=3,
            )

    mock_load_ds.assert_called_once_with("org/fraud-tabular-ds", split="test")
    assert isinstance(loader, TabularDatasetLoader)
    batch_x, batch_y, _ = next(loader.iterate_batches())
    assert batch_x.shape == (3, 2)
    assert batch_y.tolist() == [0, 1, 0]


def test_schema_mismatch_prevents_false_local_csv_pairing(tmp_path: Path):
    """Requirement 5: Matching feature count alone must NOT pair a schema-mismatched CSV."""
    wrong_csv = tmp_path / "credit_fraud_test.csv"
    df_wrong = pd.DataFrame(
        {**{f"feature_{i}": [0.1, 0.2] for i in range(30)}, "target": [0, 1]}
    )
    df_wrong.to_csv(wrong_csv, index=False)

    sentinel_raw = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    assert _schema_matches_model(wrong_csv, raw_input_features=sentinel_raw) is False

    # Matching raw schema passes and preprocesses properly
    right_csv = tmp_path / "creditcard.csv"
    df_right = pd.DataFrame(
        {**{f"V{i}": [0.1, 0.2] for i in range(1, 29)}, "Amount": [10.0, 50.0], "Time": [0.0, 3600.0], "Class": [0, 1]}
    )
    df_right.to_csv(right_csv, index=False)
    assert _schema_matches_model(right_csv, raw_input_features=sentinel_raw) is True

    expected_features = ["log_amount"] + [f"V{i}" for i in range(1, 29)] + ["hour_sin", "hour_cos"]
    loader = TabularDatasetLoader(
        data=right_csv,
        feature_names=expected_features,
        raw_input_features=sentinel_raw,
        preprocessing_info={"scaled_features": ["log_amount"], "mean": [3.0], "scale": [1.5]},
        batch_size=2,
    )
    bx, by, _ = next(loader.iterate_batches())
    assert bx.shape == (2, 31)
    assert by.tolist() == [0, 1]


def test_missing_real_dataset_keeps_ingestion_successful_and_flags_synthetic():
    """Requirements 6, 7, 8: Missing real dataset keeps ingestion successful, flags synthetic, and never pretends real evaluation."""
    from frontend.services.ingestion_service import load_model_and_dataset

    class DummyBooster:
        feature_names = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour_sin", "hour_cos"]

        def num_features(self):
            return 31

        def predict(self, dmat, output_margin=False):
            return np.zeros(len(dmat.get_data()), dtype=np.float32) + 0.1

    adapter = TreeModelAdapter(DummyBooster(), feature_names=DummyBooster.feature_names)
    meta = ModelMetadata(
        model_id="HarryDataScience/sentinel-fraud-xgboost",
        framework="xgboost",
        data_domain="tabular",
        dataset_name="custom_dataset",
        task_type="classification",
        num_classes=2,
        extra_info={"feature_names": DummyBooster.feature_names},
    )

    with patch("frontend.services.ingestion_service.ingest_model", return_value=IngestionResult(adapter=adapter, metadata=meta)):
        with patch("app.evaluation.dataset_loader.factory.resolve_tabular_dataset_source", return_value=None):
            res = load_model_and_dataset(
                model_identifier="HarryDataScience/sentinel-fraud-xgboost",
                data_domain="tabular",
                sample_size=5,
            )

    # Model ingestion succeeded
    assert res["adapter"] is not None
    assert res["test_inputs"].shape == (5, 31)
    # Clearly marked synthetic fallback — never treated as real evaluation dataset
    assert res["is_synthetic_dataset"] is True
    assert res["telemetry"]["is_synthetic_dataset"] is True
    assert res["telemetry"]["dataset_source"] == "synthetic_fallback"
    assert "TabularDatasetLoader requires 'data'" in res["dataset_fallback_reason"]



def test_xgb_model_skill_level_v2_without_dataset_no_250_fallback(tmp_path: Path):
    """Exact failure test: xgb_model_Skill_Level_v2.json with no dataset must NOT report 250 samples or treat model filename as dataset."""
    import xgboost as xgb
    from frontend.services.ingestion_service import (
        get_dataset_population_info,
        load_model_and_dataset,
    )

    model_path = tmp_path / "xgb_model_Skill_Level_v2.json"
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=(20,))
    dtrain = xgb.DMatrix(X, label=y, feature_names=["skill_a", "skill_b", "skill_c", "skill_d"])
    booster = xgb.train({"max_depth": 2, "objective": "binary:logistic"}, dtrain, num_boost_round=3)
    booster.save_model(str(model_path))

    # 1. Population info must report 0 resolved samples, never 250
    pop = get_dataset_population_info(
        dataset_name=None,
        data_domain="tabular",
        model_identifier=str(model_path),
    )
    assert pop["resolved"] is False
    assert pop["total_samples"] == 0

    # 2. Model ingestion succeeds without silent synthetic fallback (Case B)
    res = load_model_and_dataset(
        model_identifier=str(model_path),
        dataset_name=None,
        data_domain="tabular",
        framework="xgboost",
        sample_size=5,
        allow_synthetic_fallback=False,
    )
    assert res["adapter"] is not None
    assert res["dataset_resolved"] is False
    assert res["is_synthetic_dataset"] is False
    assert res["test_inputs"] is None
    assert res["total_dataset_samples"] == 0
    assert "xgb_model_Skill_Level_v2.json" not in (res["dataset_fallback_reason"] or "Could not locate tabular dataset for 'xgb_model_Skill_Level_v2.json'") or "Could not locate tabular dataset for 'xgb_model_Skill_Level_v2.json'" not in (res["dataset_fallback_reason"] or "")


def test_xgb_model_skill_level_v2_with_valid_and_reordered_csv(tmp_path: Path):
    """Case C: Local xgb_model_Skill_Level_v2.json with a valid CSV (even reordered columns) loads and runs adapter.predict."""
    import xgboost as xgb
    from frontend.services.ingestion_service import (
        get_dataset_population_info,
        load_model_and_dataset,
    )

    model_path = tmp_path / "xgb_model_Skill_Level_v2.json"
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=(20,))
    feat_names = ["skill_a", "skill_b", "skill_c", "skill_d"]
    dtrain = xgb.DMatrix(X, label=y, feature_names=feat_names)
    booster = xgb.train({"max_depth": 2, "objective": "binary:logistic"}, dtrain, num_boost_round=3)
    booster.save_model(str(model_path))

    # CSV with scrambled feature ordering
    csv_path = tmp_path / "skill_eval.csv"
    df_eval = pd.DataFrame(
        {
            "skill_d": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "skill_b": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "skill_a": [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
            "skill_c": [3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    df_eval.to_csv(csv_path, index=False)

    pop = get_dataset_population_info(dataset_name=str(csv_path), data_domain="tabular")
    assert pop["resolved"] is True
    assert pop["total_samples"] == 6

    res = load_model_and_dataset(
        model_identifier=str(model_path),
        dataset_name=str(csv_path),
        data_domain="tabular",
        framework="xgboost",
        sample_size=4,
        allow_synthetic_fallback=False,
    )
    assert res["dataset_resolved"] is True
    assert res["is_synthetic_dataset"] is False
    assert res["total_dataset_samples"] == 6
    assert res["test_inputs"].shape == (4, 4)
    # Reordered so skill_a is column 0
    assert torch.allclose(res["test_inputs"][:, 0], torch.tensor([2.1, 2.2, 2.3, 2.4]))
    preds = res["adapter"].predict(res["test_inputs"])
    assert preds.shape == (4, 2)


def test_case_d_missing_invalid_and_mismatched_datasets_raise_validation_error(tmp_path: Path):
    """Case D: Missing dataset path, invalid dataset path, and dimension mismatch must raise DatasetFormatError (not synthetic fallback)."""
    import xgboost as xgb
    from app.ingestion.exceptions import DatasetFormatError
    from frontend.services.ingestion_service import load_model_and_dataset

    model_path = tmp_path / "xgb_model_Skill_Level_v2.json"
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=(20,))
    dtrain = xgb.DMatrix(X, label=y, feature_names=["skill_a", "skill_b", "skill_c", "skill_d"])
    booster = xgb.train({"max_depth": 2, "objective": "binary:logistic"}, dtrain, num_boost_round=3)
    booster.save_model(str(model_path))

    # 1. Missing dataset path
    with pytest.raises(DatasetFormatError):
        load_model_and_dataset(
            model_identifier=str(model_path),
            dataset_name=str(tmp_path / "does_not_exist.csv"),
            data_domain="tabular",
            framework="xgboost",
            allow_synthetic_fallback=False,
        )

    # 2. Feature dimension / schema mismatch DataFrame
    bad_df = pd.DataFrame({"wrong_1": [1.0, 2.0], "wrong_2": [3.0, 4.0], "target": [0, 1]})
    with pytest.raises(DatasetFormatError):
        load_model_and_dataset(
            model_identifier=str(model_path),
            dataset_name="bad_df",
            dataset_data=bad_df,
            data_domain="tabular",
            framework="xgboost",
            allow_synthetic_fallback=False,
        )


def test_case_a_preset_benchmark_tabular_resolution():
    """Case A: Preset tabular benchmark resolves its associated dataset automatically without manual dataset path."""
    from frontend.services.ingestion_service import get_dataset_population_info

    pop = get_dataset_population_info(
        dataset_name=None,
        data_domain="tabular",
        model_identifier="tabular-mlp-adult-census",
    )
    assert pop["resolved"] is True
    assert pop["total_samples"] == 250


def test_case_e_explicit_synthetic_selection(tmp_path: Path):
    """Case E: Explicit synthetic selection marks telemetry clearly as synthetic."""
    import xgboost as xgb
    from frontend.services.ingestion_service import load_model_and_dataset

    model_path = tmp_path / "xgb_model_Skill_Level_v2.json"
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=(20,))
    dtrain = xgb.DMatrix(X, label=y, feature_names=["skill_a", "skill_b", "skill_c", "skill_d"])
    booster = xgb.train({"max_depth": 2, "objective": "binary:logistic"}, dtrain, num_boost_round=3)
    booster.save_model(str(model_path))

    res = load_model_and_dataset(
        model_identifier=str(model_path),
        dataset_name=None,
        data_domain="tabular",
        framework="xgboost",
        sample_size=5,
        allow_synthetic_fallback=True,
    )
    assert res["is_synthetic_dataset"] is True
    assert res["dataset_resolved"] is False
    assert res["telemetry"]["dataset_source"] == "synthetic_fallback"
    assert res["test_inputs"].shape == (5, 4)
