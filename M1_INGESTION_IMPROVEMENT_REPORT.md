# AdverScan M1 (Model Ingestion) Improvement & Bug Fix Report

## 1. Executive Summary

Module 1 (M1 - Model Ingestion) has been updated to provide robust, framework-aware, and domain-accurate ingestion for dynamically loaded Hugging Face models, with special emphasis on Time-Series and Tabular architectures. The silent synthetic dataset fallback behavior was eliminated in favor of transparent provenance reporting, dimension-aware shape discovery, and explicit UI gatekeeping.

All 125 existing ingestion unit tests, 62 frontend tests, and 7 newly added regression tests pass with zero regressions.

---

## 2. Root Cause Analysis of Ingestion Failures

### 2.1. Model `keras-io/timeseries_transformer_classification`
- **Hugging Face Hub Analysis**: Repository contains TensorFlow/Keras SavedModel artifacts (`saved_model.pb`, `keras_metadata.pb`, `variables/`), tagged with `tf-keras`. It does **not** contain Hugging Face Transformers `config.json` or PyTorch weights (`pytorch_model.bin`, `model.safetensors`).
- **Root Cause**:
  1. `HuggingFaceLoader` unconditionally assumed every Hugging Face Hub repository is a PyTorch Transformers model and attempted to load configuration via `AutoConfig.from_pretrained()`, failing with `OSError`.
  2. The active Python virtual environment (`.venv`) does not have `tensorflow` installed.
- **Resolution**:
  - Implemented dynamic Hugging Face repository inspection via `detect_hf_repo_info()`, checking Hub tags (`tf-keras`), libraries, and files.
  - When a Keras / TensorFlow model is identified, `HuggingFaceLoader` checks for TensorFlow availability. If missing, it raises an informative, actionable `ModelLoadError`:
    > *"Hugging Face repository 'keras-io/timeseries_transformer_classification' contains a TensorFlow/Keras SavedModel, which requires the 'tensorflow' package. TensorFlow is not installed in the current environment. Please install 'tensorflow' to ingest and evaluate TensorFlow/Keras models."*

### 2.2. Model `scikit-learn/cancer-prediction-trees`
- **Hugging Face Hub Analysis**: Repository contains `model.pkl` (a pickled Scikit-learn tree ensemble with 30 input features) and `README.md`.
- **Root Cause**:
  1. `app/ingestion/model_registry.py` had a check order where any model path detected as a Hugging Face Hub model (`detected_source == "huggingface"`) immediately assigned `framework = "huggingface"`, bypassing framework inspection and routing to `HuggingFaceLoader` instead of `SklearnLoader`.
  2. In `_generate_domain_sample_batch()`, tabular dimension generation had a hardcoded fallback of `num_features = 14`, which caused dimension mismatches against the 30-feature tree model.
- **Resolution**:
  - Updated `detect_model_framework()` and `ModelPresetRegistry.build_runtime_config()` to inspect repository files and metadata. Repositories containing `model.pkl` / `joblib` are resolved to `framework = "sklearn"`.
  - Added dynamic delegation inside `HuggingFaceLoader.load()`: if an ingested Hugging Face repository is detected as Scikit-learn, it delegates loading to `SklearnLoader(unsafe_allow_pickle=True)`.
  - Replaced hardcoded tabular dimensions with dynamic discovery from `raw_model.n_features_in_`, `feature_names_in_`, layer attributes, or metadata, raising a clear `ConfigurationError` if dimensions cannot be determined.

---

## 3. Key Improvements Implemented

### 3.1. Framework-Aware Hugging Face Inspection (`app/ingestion/utils/model_utils.py`)
- Added `detect_hf_repo_info(repo_id: str) -> Dict[str, Any]` which queries `huggingface_hub.HfApi` for repository files, library tags, and pipeline tags:
  - Framework detection: `sklearn` (`model.pkl`, `joblib`), `keras`/`tensorflow` (`saved_model.pb`, `*.h5`, `keras_metadata.pb`), `onnx` (`*.onnx`), `jax` (`flax_model.msgpack`), or `pytorch`.
  - Domain detection: `tabular`, `time-series`, `image`, or `text`.
- Updated `detect_model_framework()` to prioritize detected Hub frameworks before defaulting to `"huggingface"`.

### 3.2. Registry & Ingestion Pipeline Routing (`app/ingestion/model_registry.py`, `app/ingestion/pipeline.py`)
- `ModelPresetRegistry.build_runtime_config()` now respects detected frameworks (`sklearn`, `keras`, `tensorflow`, `onnx`, `jax`).
- `app/ingestion/pipeline.py`:
  - Added loader dispatch for `keras`, `tensorflow`, and `onnx` with explicit dependency error checks.
  - Added `_generate_validation_sample_input()` logic: when `sample_input` is not supplied by the caller, realistic domain-conforming samples are synthesized to verify the model forward-pass via `ModelValidator.validate(adapter, sample_input=validation_sample)`.

### 3.3. Forward-Pass and Model State Validation (`app/ingestion/validation/validator.py`)
- Expanded `SUPPORTED_FRAMEWORKS` to include `keras`, `tensorflow`, and `onnx`.
- Enhanced `ModelValidator.validate()`:
  - Verifies that model predictions are non-None.
  - Verifies that output shape is non-empty (`dim != 0`).
  - Verifies that predictions contain no `NaN` or `Inf` values.

### 3.4. Elimination of Silent Synthetic Fallback (`frontend/services/ingestion_service.py`, `frontend/streamlit_app.py`)
- **Dimension Discovery**: Replaced arbitrary hardcoded constants (`num_features = 14`, `seq_len = 32`, `num_channels = 1`) with deep reflection across model weights, layer shapes, config attributes (`n_features_in_`, `sequence_length`, `context_length`, `input_size`). If dimensions cannot be discovered, `ConfigurationError` is raised.
- **Provenance & Telemetry**: When a dataset cannot be loaded, `load_model_and_dataset()` records:
  - `is_synthetic_dataset = True`
  - `dataset_source = "synthetic_fallback"`
  - `dataset_fallback_reason = <actual error message>`
- **UI Warning & Confirmation Gate**:
  - In Streamlit Stage 2 (Active Model Details), a prominent critical alert is displayed detailing the failure reason, data provenance, and methodological limitations.
  - In Streamlit Stage 3 (Attack Configuration), an acknowledgment checkbox is enforced:
    > *"I acknowledge that evaluation is running on synthetic data and should only be used for pipeline testing."*
    The `Execute Adversarial Attack ⚡` button remains disabled until this confirmation is checked.

---

## 4. Verification and Test Results

### 4.1. Ingestion Test Suite
```bash
./.venv/bin/pytest tests/ingestion/ -v
Result: 125 passed, 21 warnings in 16.96s
```

### 4.2. Frontend Test Suite
```bash
./.venv/bin/pytest tests/test_frontend.py -v
Result: 62 passed, 3 warnings in 10.97s
```

### 4.3. Dynamic Hugging Face & Fallback Regression Tests
```bash
./.venv/bin/pytest tests/ingestion/test_dynamic_hf_and_fallback.py -v
Result: 7 passed, 3 warnings in 5.44s
```
- `test_detect_hf_repo_info_sklearn`: Passed
- `test_detect_hf_repo_info_keras_savedmodel`: Passed
- `test_build_runtime_config_dynamic_routing_sklearn`: Passed
- `test_hf_loader_keras_missing_tensorflow_raises_descriptive_error`: Passed
- `test_generate_domain_sample_batch_dynamic_tabular_discovery`: Passed
- `test_generate_domain_sample_batch_unknown_tabular_raises`: Passed
- `test_validator_with_non_empty_forward_pass`: Passed

### 4.4. Code Cleanliness Check
```bash
git diff --check
Result: 0 errors
```

---

## 5. Summary of Modified Files

1. [app/ingestion/utils/model_utils.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/app/ingestion/utils/model_utils.py) — Added `detect_hf_repo_info` and dynamic Hub framework inspection.
2. [app/ingestion/model_registry.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/app/ingestion/model_registry.py) — Dynamic framework routing for Hugging Face repos.
3. [app/ingestion/loader/huggingface_loader.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/app/ingestion/loader/huggingface_loader.py) — Added tabular support, sklearn delegation, and descriptive missing framework handling.
4. [app/ingestion/pipeline.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/app/ingestion/pipeline.py) — Support for Keras/TF/ONNX routing and dynamic forward-pass sample generation during validation.
5. [app/ingestion/validation/validator.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/app/ingestion/validation/validator.py) — Extended framework support and non-empty output checks.
6. [frontend/services/ingestion_service.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/frontend/services/ingestion_service.py) — Dynamic dimension discovery, removed hardcoded tabular/TS constants, explicit failure reasons.
7. [frontend/streamlit_app.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/frontend/streamlit_app.py) — Critical synthetic data alert banner and user confirmation gate on attack execution.
8. [tests/ingestion/test_validator.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/tests/ingestion/test_validator.py) — Updated validator test for extended frameworks.
9. [tests/ingestion/test_dynamic_hf_and_fallback.py](file:///home/bisag/Downloads/M.Tech_Project_Juhi/AdverScan/AdverScan/tests/ingestion/test_dynamic_hf_and_fallback.py) — New targeted regression tests for M1 dynamic ingestion.

