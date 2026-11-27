"""
Focused regression tests for the tabular evaluation preprocessing pipeline:
- Column/value whitespace stripping
- Safe numeric string coercion (including Wine Quality ';' CSV)
- Categorical encoding & missing value imputation (without dropping rows/columns)
- Diagnostics for mixed invalid values and arbitrary free text
- Consistent fitted preprocessing between evaluation and model prediction
- Original & processed feature count telemetry
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from app.evaluation.dataset_loader.tabular_loader import (
    TabularDatasetLoader,
    TabularEvaluationPreprocessor,
)
from app.ingestion.exceptions import DatasetFormatError
from frontend.services.ingestion_service import load_model_and_dataset


def test_wine_quality_with_whitespace_numeric_strings_and_missing_values(tmp_path: Path):
    """Wine Quality ';' CSV with whitespace headers, numeric strings, and missing value imputes cleanly without dropping rows."""
    wine_csv = tmp_path / "winequality-red.csv"
    wine_csv.write_text(
        '" fixed acidity ";" volatile acidity ";" citric acid ";" residual sugar ";" chlorides ";" free sulfur dioxide ";" total sulfur dioxide ";" density ";" pH ";" sulphates ";" alcohol ";" quality "\n'
        ' 7.4 ; 0.70 ; 0.00 ; 1.9 ; 0.076 ; 11.0 ; 34.0 ; 0.9978 ; 3.51 ; 0.56 ; 9.4 ; 5 \n'
        ' 7.8 ; 0.88 ;  ; 2.6 ; 0.098 ; 25.0 ; 67.0 ; 0.9968 ; 3.20 ; 0.68 ; 9.8 ; 5 \n'
        ' 11.2 ; 0.28 ; 0.56 ; 1.9 ; 0.075 ; 17.0 ; 60.0 ; 0.9980 ; 3.16 ; 0.58 ; 9.8 ; 6 \n'
        ' 7.3 ; 0.65 ; 0.00 ; 1.2 ; 0.065 ; 15.0 ; 21.0 ; 0.9946 ; 3.39 ; 0.47 ; 10.0 ; 7 \n',
        encoding="utf-8",
    )

    loader = TabularDatasetLoader(data=wine_csv, batch_size=4)
    assert loader.original_feature_count == 11
    assert loader.processed_feature_count == 11
    assert loader.target_column_name == "quality"
    assert len(loader) == 4  # Never silently drops rows
    assert loader.feature_names == [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol",
    ]
    assert "citric acid" in loader.preprocessing_summary["imputed_columns"]
    assert not torch.isnan(loader.inputs).any()
    assert loader.targets.tolist() == [5, 5, 6, 7]


def test_categorical_encoding_imputation_and_consistent_prediction(tmp_path: Path):
    """Verify categorical encoding, imputation, feature order preservation, and consistent transform on new DataFrames."""
    df_train = pd.DataFrame(
        {
            " color ": [" red ", " white ", " red ", " ", " white ", " red "],
            " acidity ": [" 7.1 ", " 6.5 ", "  ", " 8.0 ", " 6.2 ", " 7.5 "],
            " alcohol ": [9.5, 10.2, 11.0, 9.8, 12.1, 10.5],
            " quality ": [0, 1, 0, 1, 1, 0],
        }
    )
    csv_path = tmp_path / "wine_cat.csv"
    df_train.to_csv(csv_path, index=False)

    expected_order = ["alcohol", "color", "acidity"]
    loader = TabularDatasetLoader(data=csv_path, feature_names=expected_order, batch_size=6)

    assert loader.original_feature_count == 3
    assert loader.processed_feature_count == 3
    assert loader.feature_names == expected_order
    assert len(loader) == 6
    assert "color" in loader.preprocessing_summary["categorical_columns"]
    assert "acidity" in loader.preprocessing_summary["imputed_columns"]
    assert not torch.isnan(loader.inputs).any()

    # Train a sklearn model on processed evaluation features and attach the same fitted_preprocessor
    clf = LogisticRegression().fit(loader.inputs.numpy(), loader.targets.numpy())
    model_path = tmp_path / "wine_lr.joblib"
    joblib.dump(clf, model_path)

    res = load_model_and_dataset(
        model_identifier=str(model_path),
        dataset_name=str(csv_path),
        data_domain="tabular",
        framework="sklearn",
        sample_size=6,
        unsafe_allow_pickle=True,
        allow_synthetic_fallback=False,
    )
    assert res["original_feature_count"] == 3
    assert res["processed_feature_count"] == 3
    assert res["telemetry"]["original_feature_count"] == 3
    assert res["telemetry"]["processed_feature_count"] == 3
    assert res["adapter"].tabular_preprocessor is not None

    # Predicting on raw DataFrame with whitespace & missing values uses the exact same fitted preprocessor
    raw_new = pd.DataFrame({" acidity ": [" 7.1 "], " alcohol ": [9.5], " color ": [" red "]})
    preds_raw = res["adapter"].predict(raw_new)
    preds_tensor = res["adapter"].predict(res["test_inputs"][:1])
    assert preds_raw.shape == (1, 2)


def test_reject_mixed_invalid_numeric_values_with_diagnostics(tmp_path: Path):
    """Mixed numeric + invalid text must raise DatasetFormatError naming column and sample invalid values."""
    bad_csv = tmp_path / "mixed_invalid.csv"
    bad_csv.write_text(
        "density,pH,quality\n"
        "0.997,3.5,5\n"
        "not_a_number,3.2,6\n"
        "0.995,3.4,5\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetFormatError, match="density") as exc_info:
        TabularDatasetLoader(data=bad_csv)
    assert "not_a_number" in str(exc_info.value)


def test_reject_arbitrary_free_text_column_with_diagnostics(tmp_path: Path):
    """Arbitrary free-text column must raise DatasetFormatError instead of meaningless numeric encoding."""
    text_csv = tmp_path / "free_text.csv"
    rows = [
        f"7.4,\"This is a long arbitrary wine tasting note number {i} with many spaces\",5"
        for i in range(12)
    ]
    text_csv.write_text("acidity,notes,quality\n" + "\n".join(rows), encoding="utf-8")

    with pytest.raises(DatasetFormatError, match="notes") as exc_info:
        TabularDatasetLoader(data=text_csv)
    assert "arbitrary free text" in str(exc_info.value).lower()

