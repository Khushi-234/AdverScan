"""
Dataset loader for tabular / structured numerical data.
"""

from typing import Any, Generator, List, Optional, Tuple, Union
import csv
import io
import torch

from app.evaluation.dataset_loader.base_dataset_loader import BaseDatasetLoader


def detect_csv_delimiter(source: Any, candidate_delimiters: str = ";,\t|") -> str:
    """
    Detect delimiter from a file path or file-like object using csv.Sniffer
    with a deterministic fallback across candidate delimiters.
    """
    sample_text = ""
    if hasattr(source, "read"):
        pos = source.tell() if hasattr(source, "tell") else None
        raw = source.read(8192)
        if isinstance(raw, bytes):
            sample_text = raw.decode("utf-8", errors="ignore")
        else:
            sample_text = str(raw)
        if pos is not None and hasattr(source, "seek"):
            source.seek(pos)
    else:
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            sample_text = f.read(8192)

    if not sample_text.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=candidate_delimiters)
        if dialect.delimiter in candidate_delimiters:
            return dialect.delimiter
    except Exception:
        pass

    # Deterministic header-line fallback: pick delimiter yielding most columns (>=2)
    first_line = sample_text.splitlines()[0] if sample_text.splitlines() else sample_text
    best_delim = ","
    best_count = 1
    for delim in candidate_delimiters:
        try:
            row = next(csv.reader([first_line], delimiter=delim))
            if len(row) > best_count:
                best_count = len(row)
                best_delim = delim
        except Exception:
            continue
    return best_delim


def read_delimited_tabular(
    source: Any,
    delimiter: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Tuple[Any, str]:
    """
    Read a CSV/TSV/delimited file or stream into a DataFrame with automatic delimiter detection.
    Returns (DataFrame, detected_delimiter).
    """
    import pandas as pd
    import pandas.errors as pe
    from app.ingestion.exceptions import DatasetFormatError

    if delimiter is not None:
        chosen_sep = str(delimiter)
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, sep=chosen_sep, nrows=nrows)
        except (pe.EmptyDataError, pe.ParserError) as err:
            raise DatasetFormatError(
                f"Tabular dataset '{source}' is empty or malformed with delimiter {chosen_sep!r}: {err}"
            ) from err
        if nrows is None and len(df.columns) <= 1:
            raise DatasetFormatError(
                f"Tabular dataset '{source}' parsed with explicit delimiter {chosen_sep!r} produced only "
                f"{len(df.columns)} column. Cannot separate features from target label."
            )
        return df, chosen_sep

    detected = detect_csv_delimiter(source)
    tried_delims: List[str] = []
    for cand in [detected, ",", ";", "\t", "|"]:
        if cand not in tried_delims:
            tried_delims.append(cand)

    last_df = None
    last_sep = detected
    for sep in tried_delims:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df_cand = pd.read_csv(source, sep=sep, nrows=nrows)
            last_df = df_cand
            last_sep = sep
            if len(df_cand.columns) >= 2:
                return df_cand, sep
        except pe.EmptyDataError as err:
            raise DatasetFormatError(f"Tabular dataset '{source}' is empty: {err}") from err
        except Exception:
            continue

    if last_df is None:
        raise DatasetFormatError(
            f"Failed to parse tabular dataset '{source}' using delimiters {tried_delims}."
        )
    if nrows is None and len(last_df.columns) <= 1:
        raise DatasetFormatError(
            f"Tabular dataset '{source}' contains only {len(last_df.columns)} column "
            f"(detected delimiter: {last_sep!r}). Cannot separate features from target label."
        )
    return last_df, last_sep


class TabularEvaluationPreprocessor:
    """
    Reusable, stateful tabular preprocessor for evaluation and prediction.
    Normalizes headers/values, safely coerces numeric strings, rejects arbitrary text
    or corrupted mixed columns with diagnostics, imputes missing values, and encodes
    categorical columns while preserving exact feature ordering.
    """

    MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "nil", "?"}

    def __init__(self, expected_features: Optional[List[str]] = None):
        self.expected_features = [str(f).strip() for f in expected_features] if expected_features else None
        self.feature_names: List[str] = []
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.imputed_columns: List[str] = []
        self._numeric_imputer: Any = None
        self._cat_imputer: Any = None
        self._cat_encoder: Any = None
        self.is_fitted: bool = False

    @classmethod
    def normalize_dataframe(cls, df: Any) -> Any:
        """Strip column headers and whitespace on string values; normalize missing tokens."""
        import numpy as np
        import pandas as pd

        out = df.copy()
        out.columns = out.columns.astype(str).str.strip()
        for col in out.columns:
            if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]) or isinstance(out[col].dtype, pd.CategoricalDtype):
                s = out[col].astype(object)
                mask_not_na = s.notna()
                if mask_not_na.any():
                    stripped = s[mask_not_na].astype(str).str.strip()
                    is_missing_tok = stripped.str.lower().isin(cls.MISSING_TOKENS)
                    s_clean = s.copy()
                    s_clean[mask_not_na] = np.where(is_missing_tok, np.nan, stripped)
                    out[col] = s_clean
        return out

    @staticmethod
    def _is_arbitrary_free_text(non_null_series: Any) -> bool:
        """Detect unstructured free-text columns that should not be encoded as categories."""
        n = len(non_null_series)
        if n == 0:
            return False
        as_str = non_null_series.astype(str)
        avg_len = float(as_str.str.len().mean())
        max_spaces = int(as_str.str.count(r"\s+").max())
        unique_ratio = float(as_str.nunique()) / float(n)
        if avg_len > 30.0 or max_spaces >= 3:
            return True
        if n >= 10 and unique_ratio > 0.85 and avg_len > 12.0:
            return True
        return False

    def fit_transform(self, X_df: Any) -> Tuple[Any, List[str]]:
        """Validate, fit imputers/encoders, and transform X_df into float32 ndarray preserving column order."""
        import numpy as np
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OrdinalEncoder
        from app.ingestion.exceptions import DatasetFormatError

        df_norm = self.normalize_dataframe(X_df)
        if self.expected_features:
            df_norm = df_norm[self.expected_features]

        self.feature_names = list(df_norm.columns)
        self.numeric_columns = []
        self.categorical_columns = []
        self.imputed_columns = []

        for col in self.feature_names:
            series = df_norm[col]
            if pd.api.types.is_numeric_dtype(series):
                col_vals = pd.to_numeric(series, errors="coerce")
                if np.isinf(col_vals.dropna().to_numpy(dtype=float)).any():
                    bad_inf = series[np.isinf(col_vals.fillna(0).to_numpy(dtype=float))].unique()[:5].tolist()
                    raise DatasetFormatError(
                        f"Column '{col}' contains infinite values (sample: {bad_inf}) which cannot be safely processed."
                    )
                if col_vals.isna().all():
                    raise DatasetFormatError(
                        f"Column '{col}' contains only missing values and cannot be safely imputed."
                    )
                df_norm[col] = col_vals.astype(np.float64)
                self.numeric_columns.append(col)
                if col_vals.isna().any():
                    self.imputed_columns.append(col)
            else:
                non_null = series.dropna()
                if len(non_null) == 0:
                    raise DatasetFormatError(
                        f"Column '{col}' contains only missing values and cannot be safely imputed."
                    )
                parsed_num = pd.to_numeric(non_null, errors="coerce")
                n_failed = int(parsed_num.isna().sum())

                if n_failed == 0:
                    # Purely numeric strings (e.g. " 7.4 ")
                    full_num = pd.to_numeric(series, errors="coerce").astype(np.float64)
                    df_norm[col] = full_num
                    self.numeric_columns.append(col)
                    if full_num.isna().any():
                        self.imputed_columns.append(col)
                elif n_failed < len(non_null):
                    # Mixed numeric and invalid non-numeric values
                    bad_samples = non_null[parsed_num.isna()].unique()[:5].tolist()
                    raise DatasetFormatError(
                        f"Column '{col}' contains mixed numeric and invalid non-numeric values "
                        f"(sample invalid values: {bad_samples}). Fix or clean these values before evaluation."
                    )
                else:
                    # All non-numeric strings: check for arbitrary free text vs structured categories
                    if self._is_arbitrary_free_text(non_null):
                        samples = non_null.unique()[:5].tolist()
                        raise DatasetFormatError(
                            f"Column '{col}' contains arbitrary free text rather than structured categorical values "
                            f"(sample values: {samples}). Arbitrary text cannot be converted to numeric features."
                        )
                    self.categorical_columns.append(col)
                    if series.isna().any():
                        self.imputed_columns.append(col)

        # Fit imputers & encoders separately so exact original column order is preserved
        out_cols: dict = {}
        if self.numeric_columns:
            self._numeric_imputer = SimpleImputer(strategy="median")
            num_arr = self._numeric_imputer.fit_transform(df_norm[self.numeric_columns].to_numpy(dtype=np.float64))
            for idx, col in enumerate(self.numeric_columns):
                out_cols[col] = num_arr[:, idx]

        if self.categorical_columns:
            self._cat_imputer = SimpleImputer(strategy="most_frequent")
            self._cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            cat_imputed = self._cat_imputer.fit_transform(df_norm[self.categorical_columns].astype(object))
            cat_encoded = self._cat_encoder.fit_transform(cat_imputed)
            for idx, col in enumerate(self.categorical_columns):
                out_cols[col] = cat_encoded[:, idx]

        ordered_arr = np.column_stack([out_cols[col] for col in self.feature_names]).astype(np.float32)
        self.is_fitted = True
        return ordered_arr, list(self.feature_names)

    def transform(self, X_df: Any) -> Any:
        """Apply the exact same fitted preprocessing to new DataFrame inputs for consistent prediction."""
        import numpy as np
        import pandas as pd

        if not self.is_fitted:
            raise RuntimeError("TabularEvaluationPreprocessor must be fitted before calling transform().")

        df_norm = self.normalize_dataframe(X_df)
        df_norm = df_norm[self.feature_names]

        out_cols: dict = {}
        if self.numeric_columns:
            num_df = df_norm[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
            num_arr = self._numeric_imputer.transform(num_df.to_numpy(dtype=np.float64))
            for idx, col in enumerate(self.numeric_columns):
                out_cols[col] = num_arr[:, idx]

        if self.categorical_columns:
            cat_imputed = self._cat_imputer.transform(df_norm[self.categorical_columns].astype(object))
            cat_encoded = self._cat_encoder.transform(cat_imputed)
            for idx, col in enumerate(self.categorical_columns):
                out_cols[col] = cat_encoded[:, idx]

        return np.column_stack([out_cols[col] for col in self.feature_names]).astype(np.float32)


class TabularDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for tabular / structured numerical data (CSV files, DataFrames, Tensors).
    """

    def __init__(
        self,
        data: Optional[Union[str, Any]] = None,
        target_col: Optional[Union[str, int]] = "target",
        dataset_name: str = "Tabular_Dataset",
        batch_size: int = 32,
        data_path: Optional[Union[str, Any]] = None,
        target_column: Optional[Union[str, int]] = None,
        feature_names: Optional[List[str]] = None,
        raw_input_features: Optional[List[str]] = None,
        preprocessing_info: Optional[Any] = None,
        expected_num_features: Optional[int] = None,
        delimiter: Optional[str] = None,
        **kwargs: Any,
    ):
        from pathlib import Path
        import pandas as pd

        if data is None:
            data = data_path or kwargs.get("path")
        if target_column is not None:
            target_col = target_column
        if target_col is None:
            target_col = "target"

        self._dataset_name = dataset_name
        self.batch_size = batch_size
        self.detected_delimiter: Optional[str] = delimiter or kwargs.get("sep")

        df = None
        if isinstance(data, (str, Path)):
            from pathlib import Path
            p = Path(data)
            resolved_p = None
            if p.is_file():
                resolved_p = str(p)
            else:
                from app.evaluation.dataset_loader.factory import resolve_dataset_file
                resolved_p = resolve_dataset_file(
                    str(data),
                    [".csv", ".tsv", ".parquet", ".xlsx", ".json", ".jsonl", ".txt"],
                )

            if not resolved_p or not Path(resolved_p).is_file():
                raise ValueError(f"Tabular dataset file '{data}' could not be found.")

            file_path = Path(resolved_p)
            suffix = file_path.suffix.lower()

            try:
                explicit_sep = delimiter if delimiter is not None else kwargs.get("sep")
                if suffix == ".csv":
                    df, self.detected_delimiter = read_delimited_tabular(file_path, delimiter=explicit_sep)
                elif suffix == ".tsv":
                    df, self.detected_delimiter = read_delimited_tabular(file_path, delimiter=explicit_sep or "\t")
                elif suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                elif suffix == ".xlsx":
                    try:
                        import openpyxl
                    except ImportError as err:
                        from app.ingestion.exceptions import MissingDependencyError
                        raise MissingDependencyError(
                            "Reading Excel (.xlsx) tabular datasets requires 'openpyxl'. "
                            "Please install it via 'pip install openpyxl'."
                        ) from err
                    try:
                        df = pd.read_excel(file_path)
                    except Exception as err:
                        from app.ingestion.exceptions import DatasetFormatError
                        raise DatasetFormatError(f"Failed to read Excel file '{file_path}': {err}") from err
                elif suffix in (".json", ".jsonl"):
                    lines_mode = suffix == ".jsonl"
                    try:
                        df = pd.read_json(file_path, lines=lines_mode)
                    except Exception:
                        df = pd.read_json(file_path, lines=not lines_mode)
                elif suffix == ".txt":
                    # Try comma-delimited then tab-delimited
                    try:
                        df = pd.read_csv(file_path)
                    except Exception:
                        df = pd.read_csv(file_path, sep="\t")
                else:
                    raise ValueError(
                        f"Unsupported tabular file format '{suffix}'. "
                        f"Supported formats: .csv, .tsv, .parquet, .xlsx, .json, .jsonl"
                    )
            except Exception as e:
                from app.ingestion.exceptions import DatasetFormatError, MissingDependencyError
                if isinstance(e, (MissingDependencyError, DatasetFormatError)):
                    raise
                import pandas.errors as pe
                if isinstance(e, (pe.EmptyDataError, pe.ParserError)):
                    raise DatasetFormatError(f"Tabular dataset '{file_path}' is empty or malformed: {e}") from e
                raise ValueError(f"Failed to parse tabular dataset from '{file_path}': {e}") from e

        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, tuple) and len(data) == 2:
            X, y = data
            self.inputs = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
            self.targets = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
            return
        else:
            raise ValueError(
                "Data for TabularDatasetLoader must be a file path (.csv, .tsv, .parquet, .xlsx, .json, .jsonl), "
                "DataFrame, or tuple of (X, y)."
            )

        if df is not None:
            if df.empty or len(df) == 0:
                from app.ingestion.exceptions import DatasetFormatError
                raise DatasetFormatError(f"Tabular dataset from '{data}' is empty (0 samples).")

            # 1. Normalize column names and strip string values before target/feature separation
            df = TabularEvaluationPreprocessor.normalize_dataframe(df)
            if isinstance(target_col, str):
                target_col = target_col.strip()
            clean_feature_names = [str(f).strip() for f in feature_names] if feature_names else None
            clean_raw_features = [str(f).strip() for f in raw_input_features] if raw_input_features else None

            # 2. Resolve target column strictly and keep separate from input features
            resolved_col = None
            lower_col_map = {str(c).lower(): str(c) for c in df.columns}
            if target_col in df.columns:
                resolved_col = target_col
            elif target_col is not None and target_col != "target":
                raise ValueError(
                    f"Specified target column '{target_col}' not found in tabular dataset columns: {list(df.columns)}"
                )
            elif isinstance(target_col, int) and 0 <= target_col < len(df.columns):
                resolved_col = df.columns[target_col]
            else:
                for cand in ("target", "label", "class", "y", "fraud", "is_fraud", "outcome", "quality"):
                    if cand in lower_col_map:
                        resolved_col = lower_col_map[cand]
                        break
                if resolved_col is None:
                    if len(df.columns) < 2:
                        raise ValueError(
                            f"Tabular dataset contains only {len(df.columns)} column. "
                            f"Cannot separate features from target label."
                        )
                    resolved_col = df.columns[-1]
            resolved_col_str = str(resolved_col)
            self.target_column_name = resolved_col_str
            y_raw = df[resolved_col_str].values
            X_df = df.drop(columns=[resolved_col_str])

            self.original_feature_names = list(X_df.columns)
            self.original_feature_count = int(X_df.shape[1])

            # 3. Preserve feature order & apply model-specific raw-to-model transforms if needed
            if clean_feature_names:
                import numpy as np
                if set(clean_feature_names).issubset(set(X_df.columns)):
                    X_df = X_df[clean_feature_names]
                elif clean_raw_features and set(clean_raw_features).issubset(set(X_df.columns)):
                    X_proc = X_df.copy()
                    if "Amount" in X_proc.columns and "log_amount" in clean_feature_names:
                        amt = np.maximum(0.0, pd.to_numeric(X_proc["Amount"], errors="raise").astype(float).values)
                        log_amt = np.log1p(amt)
                        if isinstance(preprocessing_info, dict) and "log_amount" in preprocessing_info.get("scaled_features", []):
                            mean = float(preprocessing_info.get("mean", [0.0])[0])
                            scale = float(preprocessing_info.get("scale", [1.0])[0])
                            X_proc["log_amount"] = (log_amt - mean) / scale
                        else:
                            X_proc["log_amount"] = log_amt
                    if "Time" in X_proc.columns and "hour_sin" in clean_feature_names:
                        hour = (pd.to_numeric(X_proc["Time"], errors="raise").astype(float).values / 3600.0) % 24.0
                        X_proc["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
                        X_proc["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
                    if set(clean_feature_names).issubset(set(X_proc.columns)):
                        X_df = X_proc[clean_feature_names]
                    else:
                        missing = [c for c in clean_feature_names if c not in X_proc.columns]
                        raise ValueError(
                            f"Tabular dataset columns {list(X_df.columns)} do not match expected model features; missing: {missing}"
                        )
                else:
                    missing = [c for c in clean_feature_names if c not in X_df.columns]
                    raise ValueError(
                        f"Tabular dataset columns {list(X_df.columns)} do not match expected model features; missing: {missing}"
                    )

            # 4. Convert target labels: handle strings/categoricals or integers
            import numpy as np
            if y_raw.dtype == object or isinstance(y_raw[0] if len(y_raw) > 0 else None, str):
                y_series = pd.Series(y_raw)
                y_codes, _ = pd.factorize(y_series)
                self.targets = torch.tensor(y_codes, dtype=torch.long)
            else:
                try:
                    self.targets = torch.tensor(y_raw.astype(np.int64), dtype=torch.long)
                except Exception:
                    self.targets = torch.tensor(y_raw, dtype=torch.long)

            # 5. Fit and apply TabularEvaluationPreprocessor (numeric coercion, categorical encoding, imputation)
            preprocessor = TabularEvaluationPreprocessor(expected_features=clean_feature_names)
            X_arr, processed_names = preprocessor.fit_transform(X_df)

            if expected_num_features is not None and X_arr.shape[1] != int(expected_num_features):
                from app.ingestion.exceptions import DatasetFormatError
                raise DatasetFormatError(
                    f"Feature dimension mismatch: dataset provides {X_arr.shape[1]} features, "
                    f"but model expects {int(expected_num_features)} features."
                )

            self.fitted_preprocessor = preprocessor
            self.feature_names = processed_names
            self.processed_feature_count = int(X_arr.shape[1])
            self.preprocessing_summary = {
                "original_feature_count": self.original_feature_count,
                "processed_feature_count": self.processed_feature_count,
                "numeric_columns": list(preprocessor.numeric_columns),
                "categorical_columns": list(preprocessor.categorical_columns),
                "imputed_columns": list(preprocessor.imputed_columns),
                "target_column": self.target_column_name,
            }
            self.inputs = torch.tensor(X_arr, dtype=torch.float32)

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def __len__(self) -> int:
        return len(self.inputs)

    def iterate_batches(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[int]], None, None]:
        total_samples = len(self.inputs)
        for i in range(0, total_samples, self.batch_size):
            batch_inputs = self.inputs[i : i + self.batch_size]
            batch_targets = self.targets[i : i + self.batch_size]
            targets_list = batch_targets.cpu().numpy().tolist()
            yield batch_inputs, batch_targets, targets_list
