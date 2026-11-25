"""
Regression tests for delimiter-aware tabular CSV parsing (Wine Quality ';', ',', tab, '|', quoted values).
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.evaluation.dataset_loader.factory import get_dataset_loader
from app.evaluation.dataset_loader.tabular_loader import (
    TabularDatasetLoader,
    detect_csv_delimiter,
    read_delimited_tabular,
)
from app.ingestion.adapters.sklearn_adapter import SklearnAdapter
from app.ingestion.exceptions import DatasetFormatError


WINE_QUALITY_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def test_wine_quality_semicolon_csv_and_sklearn_pipeline_compatibility(tmp_path: Path):
    """Wine Quality CSV with ';' produces 11 numerical features + 'quality' target and works with sklearn Pipeline."""
    wine_csv = tmp_path / "winequality-red.csv"
    header = ";".join(f'"{c}"' for c in WINE_QUALITY_FEATURES + ["quality"])
    rows = [
        "7.4;0.7;0.0;1.9;0.076;11.0;34.0;0.9978;3.51;0.56;9.4;5",
        "7.8;0.88;0.0;2.6;0.098;25.0;67.0;0.9968;3.2;0.68;9.8;5",
        "11.2;0.28;0.56;1.9;0.075;17.0;60.0;0.998;3.16;0.58;9.8;6",
        "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0;7",
    ]
    wine_csv.write_text("\n".join([header] + rows), encoding="utf-8")

    # 1. Raw DataFrame parsing preserves all 12 columns
    df, delim = read_delimited_tabular(wine_csv)
    assert delim == ";"
    assert list(df.columns) == WINE_QUALITY_FEATURES + ["quality"]
    assert df.shape == (4, 12)

    # 2. TabularDatasetLoader separates 11 features in exact order and 'quality' target
    loader = get_dataset_loader(
        dataset_name=str(wine_csv),
        data_domain="tabular",
        feature_names=WINE_QUALITY_FEATURES,
        batch_size=4,
    )
    assert isinstance(loader, TabularDatasetLoader)
    assert loader.detected_delimiter == ";"
    assert loader.feature_names == WINE_QUALITY_FEATURES

    bx, by, _ = next(loader.iterate_batches())
    assert bx.shape == (4, 11)
    assert torch.isclose(bx[0, 0], torch.tensor(7.4))
    assert torch.isclose(bx[2, 10], torch.tensor(9.8))
    assert by.tolist() == [5, 5, 6, 7]

    # 3. Compatible with fitted scikit-learn Pipeline + SklearnAdapter
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(bx.numpy(), np.array([0, 0, 1, 1]))
    adapter = SklearnAdapter(pipe)
    preds = adapter.predict(bx)
    assert preds.shape == (4, 2)


def test_standard_comma_and_tab_and_pipe_delimiters(tmp_path: Path):
    """Verify automatic detection for ',', '\\t', and '|' delimiters."""
    # Comma
    comma_csv = tmp_path / "comma.csv"
    comma_csv.write_text("f1,f2,target\n1.0,2.0,0\n3.0,4.0,1\n", encoding="utf-8")
    l_comma = TabularDatasetLoader(data=comma_csv)
    assert l_comma.detected_delimiter == ","
    assert l_comma.inputs.shape == (2, 2)

    # Tab
    tsv_file = tmp_path / "tab_data.tsv"
    tsv_file.write_text("f1\tf2\ttarget\n1.0\t2.0\t0\n3.0\t4.0\t1\n", encoding="utf-8")
    l_tab = TabularDatasetLoader(data=tsv_file)
    assert l_tab.detected_delimiter == "\t"
    assert l_tab.inputs.shape == (2, 2)

    # Pipe
    pipe_csv = tmp_path / "pipe.csv"
    pipe_csv.write_text("f1|f2|target\n1.0|2.0|0\n3.0|4.0|1\n", encoding="utf-8")
    l_pipe = TabularDatasetLoader(data=pipe_csv)
    assert l_pipe.detected_delimiter == "|"
    assert l_pipe.inputs.shape == (2, 2)


def test_csv_with_quoted_values(tmp_path: Path):
    """Verify CSV with quoted headers and values (including embedded delimiter) parses cleanly."""
    quoted_csv = tmp_path / "quoted.csv"
    quoted_csv.write_text(
        '"feat;one";"feat two";"label"\n1.5;2.5;0\n3.5;4.5;1\n',
        encoding="utf-8",
    )
    loader = TabularDatasetLoader(data=quoted_csv)
    assert loader.detected_delimiter == ";"
    assert loader.feature_names == ["feat;one", "feat two"]
    assert loader.inputs.shape == (2, 2)
    assert loader.targets.tolist() == [0, 1]


def test_incorrect_explicit_delimiter_raises_clear_error(tmp_path: Path):
    """Verify explicit wrong delimiter raises DatasetFormatError showing delimiter and single-column failure."""
    wine_csv = tmp_path / "wine.csv"
    wine_csv.write_text("a;b;quality\n1.0;2.0;5\n3.0;4.0;6\n", encoding="utf-8")

    with pytest.raises(DatasetFormatError, match="produced only 1 column"):
        TabularDatasetLoader(data=wine_csv, delimiter=",")


def test_missing_target_column_and_single_column_csv_raise_error(tmp_path: Path):
    """Verify single-column CSV and missing explicit target column raise clear errors (never synthetic fallback)."""
    single_col = tmp_path / "single.csv"
    single_col.write_text("only_col\n1.0\n2.0\n", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="contains only 1 column"):
        TabularDatasetLoader(data=single_col)

    missing_tgt = tmp_path / "no_tgt.csv"
    missing_tgt.write_text("a;b;c\n1.0;2.0;3.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Specified target column 'quality' not found"):
        TabularDatasetLoader(data=missing_tgt, target_col="quality")

