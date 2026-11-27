"""
Factory function to construct domain-specific dataset loaders for evaluation.
"""

from typing import Any, Optional

from app.evaluation.dataset_loader.base_dataset_loader import BaseDatasetLoader
from app.evaluation.dataset_loader.vision_loader import HFVisionDatasetLoader
from app.evaluation.dataset_loader.text_loader import HFTextDatasetLoader
from app.evaluation.dataset_loader.timeseries_loader import TimeSeriesDatasetLoader
from app.evaluation.dataset_loader.tabular_loader import TabularDatasetLoader
from app.evaluation.dataset_loader.generic_loader import GenericDatasetLoader


def resolve_dataset_file(
    identifier: Optional[str],
    candidate_extensions: Any,
    model_identifier: Optional[str] = None,
) -> Optional[str]:
    """
    Dynamically locate a dataset file on disk without hardcoding any model or dataset names.
    Searches for the identifier directly, then in data/ and datasets/ directories,
    and optionally checks the model's stem name as fallback.
    """
    from pathlib import Path
    import re

    if not identifier and not model_identifier:
        return None

    search_dirs = [Path("data"), Path("datasets"), Path(".")]
    candidates_to_try = []

    if identifier:
        p = Path(identifier)
        if p.is_file():
            return str(p)
        clean_id = p.stem.lower().replace("-", "_")
        candidates_to_try.extend([
            clean_id,
            f"{clean_id}_test",
            f"{clean_id}_data",
            clean_id.replace("_", "-"),
            f"{clean_id.replace('_', '-')}_test",
        ])

    if model_identifier:
        m_p = Path(model_identifier)
        clean_m = m_p.stem.lower().replace("-", "_")
        stripped_suffix = re.sub(r"([_-]+(?:classifier|model|mlp|net|pt|pth))+$", "", clean_m)
        stripped_prefix = re.sub(r"^((?:tabular|time_series|time|series|ts|model|mlp)[_-]+)+", "", clean_m)
        stripped_both = re.sub(r"^((?:tabular|time_series|time|series|ts|model|mlp)[_-]+)+", "", stripped_suffix)
        tokens = [t for t in clean_m.split("_") if t not in ("model", "classifier", "net", "mlp")]
        model_candidates = {clean_m, stripped_suffix, stripped_prefix, stripped_both, "_".join(tokens)}

        for cand in list(model_candidates):
            if cand:
                candidates_to_try.extend([
                    cand,
                    f"{cand}_test",
                    f"{cand}_data",
                    cand.replace("_", "-"),
                    f"{cand.replace('_', '-')}-test",
                ])

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for ext in candidate_extensions:
            for name in candidates_to_try:
                candidate = s_dir / f"{name}{ext}"
                if candidate.is_file():
                    return str(candidate)

    return None


def _schema_matches_model(
    data_source: Any,
    feature_names: Optional[Any] = None,
    raw_input_features: Optional[Any] = None,
) -> bool:
    """
    Verify that a candidate tabular dataset actually matches the model's expected feature schema
    when the model declares named features, rather than assuming column count equivalence.
    """
    expected = list(feature_names) if feature_names else None
    raw_expected = list(raw_input_features) if raw_input_features else None
    if not expected and not raw_expected:
        return True

    try:
        from pathlib import Path
        import pandas as pd

        cols: Optional[set] = None
        if isinstance(data_source, pd.DataFrame):
            cols = set(data_source.columns)
        elif isinstance(data_source, (str, Path)) and Path(data_source).is_file():
            p = Path(data_source)
            if p.suffix.lower() in (".csv", ".tsv"):
                from app.evaluation.dataset_loader.tabular_loader import read_delimited_tabular
                df_hdr, _ = read_delimited_tabular(
                    p,
                    delimiter="\t" if p.suffix.lower() == ".tsv" else None,
                    nrows=2,
                )
                cols = set(df_hdr.columns)
            elif p.suffix.lower() == ".parquet":
                cols = set(pd.read_parquet(p).columns)

        if cols is None:
            return True
        if expected and set(expected).issubset(cols):
            return True
        if raw_expected and set(raw_expected).issubset(cols):
            return True
        return False
    except Exception:
        return True


def _load_hf_dataset_as_dataframe(ds_identifier: str, split: str = "test") -> Optional[Any]:
    """Attempt to load a Hugging Face dataset identifier into a Pandas DataFrame."""
    if not ds_identifier or "/" not in ds_identifier and ds_identifier.lower() in ("custom_dataset", "evaluation-dataset"):
        return None
    try:
        from datasets import load_dataset
        clean_split = split.split("[")[0] if isinstance(split, str) and "[" in split else (split or "test")
        try:
            ds = load_dataset(ds_identifier, split=clean_split)
        except Exception:
            ds = load_dataset(ds_identifier, split="train")
        return ds.to_pandas()
    except Exception:
        return None


def resolve_tabular_dataset_source(
    dataset_name: Optional[str],
    model_id: Optional[str] = None,
    split: str = "test",
    feature_names: Optional[Any] = None,
    raw_input_features: Optional[Any] = None,
) -> Optional[Any]:
    """
    Resolve a real tabular evaluation dataset (file path or DataFrame) using:
    1. Explicit local file path or dataset name
    2. Dataset files bundled inside a Hugging Face model repository
    3. Dataset identifiers declared in HF model card / metadata.json
    4. Benchmark / model identifier local file mapping (with feature schema verification)
    """
    from pathlib import Path

    tabular_exts = [".csv", ".tsv", ".parquet", ".xlsx", ".json", ".jsonl"]
    is_placeholder = not dataset_name or str(dataset_name).strip().lower() in (
        "", "custom_dataset", "evaluation-dataset", "none"
    )

    # 1. Explicit dataset_name (local file or HF dataset) — never treat a model artifact file as dataset
    _model_artifact_exts = (".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".pkl", ".pickle", ".joblib", ".onnx", ".ubj", ".cbm", ".h5", ".hdf5", ".keras", ".tflite", ".xml", ".engine", ".plan", ".mlmodel")
    is_model_file_ref = bool(dataset_name and str(dataset_name).strip().lower().endswith(_model_artifact_exts))
    if not is_placeholder and dataset_name and not is_model_file_ref and str(dataset_name).strip() != str(model_id or "").strip():
        local_match = resolve_dataset_file(dataset_name, tabular_exts, model_identifier=model_id)
        if local_match and _schema_matches_model(local_match, feature_names, raw_input_features):
            return local_match
        if "/" in str(dataset_name) and not str(dataset_name).startswith((".", "/")):
            df = _load_hf_dataset_as_dataframe(str(dataset_name), split=split)
            if df is not None and _schema_matches_model(df, feature_names, raw_input_features):
                return df

    # 2. Hugging Face model repository inspection (bundled dataset files & metadata candidates)
    if model_id and "/" in str(model_id) and not str(model_id).startswith((".", "/")):
        try:
            from app.ingestion.utils.model_utils import detect_hf_repo_info
            from huggingface_hub import hf_hub_download

            hf_info = detect_hf_repo_info(str(model_id))
            ds_files = list(hf_info.get("dataset_files", []) or [])
            if dataset_name and not is_placeholder:
                ds_key = str(dataset_name).strip().lower()
                ds_files.sort(key=lambda f: (0 if str(f).lower() == ds_key or str(f).lower().endswith("/" + ds_key) else 1))
            for ds_file in ds_files:
                try:
                    local_ds = hf_hub_download(str(model_id), ds_file)
                    if local_ds and _schema_matches_model(local_ds, feature_names, raw_input_features):
                        return local_ds
                except Exception:
                    continue

            for cand_id in hf_info.get("dataset_candidates", []) or []:
                local_cand = resolve_dataset_file(cand_id, tabular_exts, model_identifier=model_id)
                if local_cand and _schema_matches_model(local_cand, feature_names, raw_input_features):
                    return local_cand
                if "/" in cand_id:
                    df = _load_hf_dataset_as_dataframe(cand_id, split=split)
                    if df is not None and _schema_matches_model(df, feature_names, raw_input_features):
                        return df
        except Exception:
            pass

    # 3. Local model_id / benchmark mapping (only if schema matches)
    if model_id:
        local_model_ds = resolve_dataset_file(None, tabular_exts, model_identifier=str(model_id))
        if local_model_ds and _schema_matches_model(local_model_ds, feature_names, raw_input_features):
            return local_model_ds

    return None


def _load_hf_timeseries_source(
    dataset_name: Optional[str],
    model_id: Optional[str] = None,
    split: str = "test",
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Resolve a time-series dataset from bundled HF model repo files or a Hugging Face
    dataset identifier (`owner/dataset`) via `datasets.load_dataset`.
    Returns `(inputs, targets)` or `(None, None)` if unavailable.
    """
    ts_exts = (".pt", ".pth", ".npz", ".npy", ".csv", ".tsv", ".parquet")
    is_placeholder = not dataset_name or str(dataset_name).strip().lower() in (
        "", "custom_dataset", "evaluation-dataset", "none"
    )

    # 1. Check HF model repo for bundled time-series files
    if model_id and "/" in str(model_id) and not str(model_id).startswith((".", "/")):
        try:
            from app.ingestion.utils.model_utils import detect_hf_repo_info
            from huggingface_hub import hf_hub_download

            hf_info = detect_hf_repo_info(str(model_id))
            ds_files = list(hf_info.get("dataset_files", []) or [])
            if dataset_name and not is_placeholder:
                ds_key = str(dataset_name).strip().lower()
                ds_files.sort(key=lambda f: (0 if str(f).lower() == ds_key or str(f).lower().endswith("/" + ds_key) else 1))
            for ds_file in ds_files:
                if str(ds_file).lower().endswith(ts_exts):
                    try:
                        local_ds = hf_hub_download(str(model_id), ds_file)
                        if local_ds:
                            return local_ds, None
                    except Exception:
                        continue
        except Exception:
            pass

    # 2. Load Hugging Face Hub dataset (owner/dataset)
    if not is_placeholder and dataset_name and "/" in str(dataset_name) and not str(dataset_name).startswith((".", "/")):
        df = _load_hf_dataset_as_dataframe(str(dataset_name), split=split)
        if df is not None and not df.empty:
            import numpy as np

            target_candidates = ["target", "label", "labels", "class", "y"]
            target_col = next((c for c in target_candidates if c in df.columns), None)
            targets = df[target_col].to_numpy() if target_col else None
            feat_df = df.drop(columns=[target_col]) if target_col else df

            # Handle sequence-valued columns (e.g. list/array per row) or numeric tabular columns
            seq_cols = [
                c for c in feat_df.columns
                if len(feat_df) > 0 and isinstance(feat_df[c].iloc[0], (list, tuple, np.ndarray))
            ]
            if seq_cols:
                arrays = [np.stack([np.asarray(v, dtype=np.float32) for v in feat_df[c]]) for c in seq_cols]
                inputs = arrays[0] if len(arrays) == 1 else np.stack(arrays, axis=-1)
                return inputs, targets

            num_df = feat_df.select_dtypes(include=["number"])
            if not num_df.empty:
                return num_df.to_numpy(dtype=np.float32), targets

    return None, None


def get_dataset_loader(
    dataset_name: str,
    data_domain: str = "image",
    processor_name: Optional[str] = None,
    split: str = "test",
    batch_size: int = 32,
    **kwargs: Any,
) -> BaseDatasetLoader:
    """
    Factory function to construct domain-specific dataset loaders for evaluation.
    Supports data_domain: 'image'/'vision', 'text'/'nlp', 'time_series'/'timeseries'/'sequence',
    'tabular'/'csv'/'table', 'generic'/'custom'.
    """
    domain_clean = data_domain.strip().lower().replace("-", "_")

    # Guard: Auto-route known text datasets if inadvertently passed with data_domain="image"
    known_text_datasets = {
        "sst2", "stanfordnlp/sst2", "glue/sst2", "tweet_eval",
        "cardiffnlp/tweet_eval", "ag_news", "imdb", "rotten_tomatoes"
    }
    clean_ds_key = dataset_name.strip().lower() if isinstance(dataset_name, str) else ""
    if clean_ds_key in ("sst2", "glue/sst2"):
        dataset_name = "stanfordnlp/sst2"

    if clean_ds_key in known_text_datasets and domain_clean in ("image", "vision"):
        domain_clean = "text"

    _tabular_only_keys = {
        "data",
        "model_id",
        "feature_names",
        "raw_input_features",
        "preprocessing_info",
        "expected_num_features",
    }

    if domain_clean in ("image", "vision"):
        vision_kwargs = {k: v for k, v in kwargs.items() if k not in _tabular_only_keys}
        return HFVisionDatasetLoader(
            dataset_name=dataset_name,
            processor_name=processor_name,
            split=split,
            batch_size=batch_size,
            **vision_kwargs,
        )
    elif domain_clean in ("text", "nlp"):
        candidate_id = kwargs.get("data") or dataset_name
        resolved_file = resolve_dataset_file(
            candidate_id, [".txt", ".csv", ".tsv", ".json", ".jsonl"]
        )
        target_name = resolved_file or dataset_name
        text_kwargs = {k: v for k, v in kwargs.items() if k not in _tabular_only_keys}
        return HFTextDatasetLoader(
            dataset_name=target_name,
            tokenizer_name=processor_name,
            split=split,
            batch_size=batch_size,
            **text_kwargs,
        )
    elif domain_clean in ("time_series", "timeseries", "sequence"):
        inputs = kwargs.get("inputs")
        targets = kwargs.get("targets")

        if inputs is None:
            # Dynamically resolve file from data arg or dataset_name
            candidate_id = kwargs.get("data") or dataset_name
            model_id = kwargs.get("model_name") or kwargs.get("model_id")
            resolved_file = resolve_dataset_file(
                candidate_id,
                [".pt", ".pth", ".npz", ".npy", ".csv", ".tsv", ".parquet"],
                model_identifier=model_id,
            )
            if resolved_file:
                inputs = resolved_file

        if inputs is None:
            hf_inputs, hf_targets = _load_hf_timeseries_source(
                dataset_name=dataset_name,
                model_id=kwargs.get("model_name") or kwargs.get("model_id"),
                split=split,
            )
            if hf_inputs is not None:
                inputs = hf_inputs
                if targets is None:
                    targets = hf_targets

        if inputs is None:
            raise ValueError(
                f"TimeSeriesDatasetLoader requires 'inputs' and 'targets' keyword arguments or a resolvable time-series file/HF dataset for '{dataset_name}'."
            )
        return TimeSeriesDatasetLoader(
            inputs=inputs,
            targets=targets,
            dataset_name=dataset_name,
            batch_size=batch_size,
        )
    elif domain_clean in ("tabular", "csv", "table"):
        data_source = kwargs.get("data")
        feature_names = kwargs.get("feature_names")
        raw_input_features = kwargs.get("raw_input_features")
        preprocessing_info = kwargs.get("preprocessing_info")
        if data_source is None:
            model_id = kwargs.get("model_name") or kwargs.get("model_id") or processor_name
            data_source = resolve_tabular_dataset_source(
                dataset_name=dataset_name,
                model_id=model_id,
                split=split,
                feature_names=feature_names,
                raw_input_features=raw_input_features,
            )

        if data_source is None:
            raise ValueError(
                f"TabularDatasetLoader requires 'data' keyword argument (file path or DataFrame). "
                f"Could not locate tabular dataset for '{dataset_name}'."
            )
        return TabularDatasetLoader(
            data=data_source,
            target_col=kwargs.get("target_col", "target"),
            dataset_name=dataset_name,
            batch_size=batch_size,
            feature_names=feature_names,
            raw_input_features=raw_input_features,
            preprocessing_info=preprocessing_info,
            expected_num_features=kwargs.get("expected_num_features"),
            delimiter=kwargs.get("delimiter"),
        )
    elif domain_clean in ("generic", "custom"):
        if "inputs" not in kwargs or "targets" not in kwargs:
            raise ValueError(
                "GenericDatasetLoader requires both 'inputs' and 'targets' keyword arguments."
            )
        return GenericDatasetLoader(
            inputs=kwargs["inputs"],
            targets=kwargs["targets"],
            dataset_name=dataset_name,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"Unsupported data domain: '{data_domain}'. "
            f"Expected one of: 'image', 'vision', 'text', 'nlp', 'time_series', 'timeseries', 'sequence', 'tabular', 'csv', 'table', 'generic', 'custom'."
        )
