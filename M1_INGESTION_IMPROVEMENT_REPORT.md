# AdverScan M1 (Model Ingestion) Improvement & Bug Fix Report

## 1. Summary Table

| Improvement | Status | Files Modified | Test Result |
|---|---|---|---|
| Priority 1: HF Repo Inspection & Framework Routing | Implemented | `app/ingestion/utils/model_utils.py`, `app/ingestion/loader/huggingface_loader.py`, `app/ingestion/model_registry.py` | Passed (unit + mocked Hub) |
| Priority 2: Framework Loaders & Adapters (PyTorch, Sklearn, Keras/TF, ONNX, JAX) | Implemented | `app/ingestion/adapters/onnx_adapter.py`, `app/ingestion/adapters/keras_adapter.py`, `app/ingestion/adapters/__init__.py`, `app/ingestion/pipeline.py`, `app/ingestion/loader/sklearn_loader.py` | Passed (135 passed, 2 skipped in `tests/ingestion/`) |
| Priority 3 & 5: Forward-Pass Validation & Dimension Discovery | Implemented | `app/ingestion/pipeline.py`, `app/ingestion/validation/validator.py` | Passed |
| Priority 4: Synthetic Dataset Fallback Warning & Gate | Blocked on current branch (`frontend/` is on `development`, not `feature/model-ingestion`) | N/A on `feature/model-ingestion` | 2 frontend tests skipped on this branch |
| Priority 6: Standardized Input Adapters | Implemented | `app/ingestion/adapters/onnx_adapter.py`, `app/ingestion/adapters/keras_adapter.py`, `app/ingestion/pipeline.py` | Passed |
| Priority 7: Scikit-learn Preprocessing Inspection | Implemented | `app/ingestion/loader/sklearn_loader.py`, `app/ingestion/pipeline.py` | Passed |
| Priority 8 & 9: Targeted Tests & Regression Validation | Implemented & Executed | `tests/ingestion/test_dynamic_hf_and_fallback.py`, `tests/ingestion/test_validator.py` | `tests/ingestion/`: 135 passed, 2 skipped |

---

## 2. Root Causes of Both Hugging Face Ingestion Errors

1. **`keras-io/timeseries_transformer_classification`**:
   - The Hugging Face repository contains TensorFlow/Keras SavedModel artifacts (`saved_model.pb`, `keras_metadata.pb`, `variables/`) and is tagged `tf-keras`, without `config.json` or PyTorch weights.
   - `HuggingFaceLoader` previously assumed all Hugging Face repositories were `transformers` PyTorch models and failed calling `AutoConfig.from_pretrained()`. Furthermore, `tensorflow` is not installed in `.venv`.
   - **Fix**: `detect_hf_repo_info()` inspects repo files, `README.md` YAML frontmatter, `config.json`, and Hub metadata to detect `keras`/`tensorflow` and `time-series`. `HuggingFaceLoader` and `pipeline.py` route Keras/TF models cleanly and raise an actionable `ModelLoadError` when `tensorflow` is not installed (or wrap in `KerasAdapter` when available).

2. **`scikit-learn/cancer-prediction-trees`**:
   - The Hugging Face repository contains `model.pkl` (a pickled scikit-learn tree ensemble) rather than a `transformers` model.
   - `ModelPresetRegistry.build_runtime_config()` previously mapped any Hugging Face Hub identifier to `framework = "huggingface"`, bypassing `SklearnLoader`.
   - **Fix**: `detect_hf_repo_info()` detects `.pkl`/`.joblib`/`.pickle` artifacts and `sklearn` metadata, `build_runtime_config()` routes to `framework = "sklearn"`, and `SklearnLoader` locates the exact Hub artifact (`model_file`) while enforcing `unsafe_allow_pickle` consent.

---

## 3. Improvements Implemented

- **Hugging Face Repository Classification (`app/ingestion/utils/model_utils.py`)**:
  - `detect_hf_repo_info()` inspects `HfApi.list_repo_files()`, `HfApi.model_info()`, `README.md` YAML frontmatter (`_parse_yaml_frontmatter()`), and `config.json`.
  - Distinguishes `repo_type` across `"model"` (has actual serialized weights), `"dataset"` (`.parquet`/`.arrow`/`dataset_info.json`), `"source_only"` (`.py`/`.ipynb` only), `"incomplete"` (metadata/config without weights), and `"not_found"`.
- **Framework-Specific Loaders & Adapters**:
  - Added [onnx_adapter.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/app/ingestion/adapters/onnx_adapter.py) (`ONNXAdapter`) and [keras_adapter.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/app/ingestion/adapters/keras_adapter.py) (`KerasAdapter`), exported in [__init__.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/app/ingestion/adapters/__init__.py).
  - Updated [pipeline.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/app/ingestion/pipeline.py) to support ONNX (local and HF Hub `.onnx` download), Keras/TF, Sklearn (restoring `unsafe_allow_pickle=False` default security boundary), and JAX/Flax (`UnsupportedModelError`).
- **Scikit-learn Artifact Discovery & Preprocessing Inspection (`app/ingestion/loader/sklearn_loader.py`)**:
  - Uses `detect_hf_repo_info()` to discover arbitrary `.joblib`/`.pkl`/`.pickle` filenames on HF Hub.
  - Added `SklearnLoader.inspect_preprocessing()` to distinguish `Pipeline` / `ColumnTransformer` models with attached preprocessing from raw estimators lacking preprocessing, recording warnings and step details in `LoadResult.evidence` and `runtime_metadata.extra_info["preprocessing"]`.
- **Forward-Pass Validation & Dimension Discovery (`app/ingestion/pipeline.py`, `app/ingestion/validation/validator.py`)**:
  - Generates representative `validation_sample` inputs across `tabular` (including PyTorch `Linear` layer inspection), `time-series`, `image`, and `text` domains when `sample_input=None`, validating non-None, non-empty shape, and finite (non-NaN/Inf) outputs.

---

## 4. Test Results

- `tests/ingestion/`: **135 passed, 2 skipped** (`test_generate_domain_sample_batch_*` skipped because `frontend.services.ingestion_service` is not on the `feature/model-ingestion` branch).
- `tests/vulnerability_analysis/` + `tests/attack_engine/`: **28 passed, 1 failed** (`tests/attack_engine/test_pgd.py::test_pgd_registration` failed pre-existing assertion `assert 'pgm' in ['deepfool', 'fgsm', 'pgd']`, unrelated to M1).
- `tests/orchestration/test_orchestrator.py`: Collection error (`ImportError: cannot import name 'AdverScanOrchestrator'`) on `feature/model-ingestion` branch (orchestration module lives on `Orchestration`/`development`).
- `git diff --check`: 0 whitespace/syntax errors.

---

## 5. Remaining Unsupported Formats & Blocked Functionality

- **Unsupported / Missing Optional Dependencies in `.venv`**:
  - `tensorflow` / `keras` and `onnxruntime` are not installed in `.venv`; models in these formats raise descriptive `ModelLoadError`s unless the optional package is installed.
  - `jax` / `flax` models raise `UnsupportedModelError`.
- **Blocked on Current Branch (`feature/model-ingestion`)**:
  - **Priority 4 (Frontend Synthetic Dataset Fallback UI & `frontend/services/ingestion_service.py`)**: Those frontend files live on the `development` branch, not `feature/model-ingestion`. Per branch-preservation rules, `development` was not modified from this branch.
