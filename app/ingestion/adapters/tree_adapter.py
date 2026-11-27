"""
Tree-based model adapter implementation for standardizing XGBoost, LightGBM, and CatBoost models.
Exposes a unified prediction interface compatible with AdverScan's evaluation and attack engines.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

from app.ingestion.adapters.base_adapter import AdapterCapabilities, BaseModelAdapter


class TreeModelAdapter(BaseModelAdapter):
    """
    Standardized adapter for tree-based models:
    - XGBoost (Booster, XGBClassifier)
    - LightGBM (Booster, LGBMClassifier)
    - CatBoost (CatBoostClassifier)

    Correctly flags supports_gradients=False so that continuous gradient attacks
    are rejected cleanly while black-box, decision-based, and scoring evaluations operate properly.
    """

    def __init__(
        self,
        model: Any,
        device: Union[str, torch.device] = "cpu",
        processor: Optional[Any] = None,
        data_domain: str = "tabular",
        framework: str = "tree",
        feature_names: Optional[List[str]] = None,
        raw_input_features: Optional[List[str]] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        calibrator_info: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
    ):
        self._model = model
        self._device = str(device)
        self.processor = processor
        self.data_domain = data_domain
        self.framework = framework

        # Feature and calibration metadata
        self.feature_names: Optional[List[str]] = feature_names or getattr(model, "feature_names", None)
        if self.feature_names is not None:
            self.feature_names = list(self.feature_names)

        self.raw_input_features: Optional[List[str]] = raw_input_features or getattr(model, "raw_input_features", None)
        self.preprocessing_info: Optional[Dict[str, Any]] = preprocessing_info or getattr(model, "preprocessing_info", None)
        self.calibrator_info: Optional[Dict[str, Any]] = calibrator_info or getattr(model, "calibrator_info", None)
        self.threshold: float = getattr(model, "threshold", threshold)

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Tree models are non-differentiable in continuous autograd."""
        return AdapterCapabilities(
            supports_gradients=False,
            supports_predict=True,
            supports_batch_predict=True,
            supports_logits=True,
        )

    @property
    def device(self) -> str:
        return self._device

    def get_model(self) -> Any:
        return self._model

    def to(self, device: Any) -> "TreeModelAdapter":
        self._device = str(device)
        return self

    def eval(self) -> "TreeModelAdapter":
        return self

    def train(self, mode: bool = True) -> "TreeModelAdapter":
        return self

    def _to_numpy(self, inputs: Any) -> np.ndarray:
        if isinstance(inputs, torch.Tensor):
            return inputs.detach().cpu().numpy().astype(np.float32)
        if isinstance(inputs, np.ndarray):
            return inputs.astype(np.float32)
        if hasattr(inputs, "values"):
            return np.asarray(inputs.values, dtype=np.float32)
        return np.asarray(inputs, dtype=np.float32)

    def _preprocess_dataframe(self, df: Any) -> Any:
        """Apply preprocessing transformations if DataFrame contains raw input features."""
        if not hasattr(df, "columns"):
            return df

        fitted_prep = getattr(self, "tabular_preprocessor", None)
        if fitted_prep is not None and getattr(fitted_prep, "is_fitted", False):
            return fitted_prep.transform(df)

        cols = list(df.columns)
        # If columns already match expected model feature names, reorder and return
        if self.feature_names and set(self.feature_names).issubset(set(cols)):
            return df[self.feature_names]

        # If columns match raw features and preprocessing info is available
        if self.preprocessing_info and self.raw_input_features and set(self.raw_input_features).issubset(set(cols)):
            import pandas as pd
            df_proc = df.copy()

            # Handle Sentinel-style log_amount and hour sin/cos if present
            if "Amount" in cols and "log_amount" in (self.feature_names or []):
                amt = np.maximum(0.0, df_proc["Amount"].astype(float).values)
                log_amt = np.log1p(amt)
                scaled_feats = self.preprocessing_info.get("scaled_features", [])
                if "log_amount" in scaled_feats:
                    mean = float(self.preprocessing_info.get("mean", [0.0])[0])
                    scale = float(self.preprocessing_info.get("scale", [1.0])[0])
                    df_proc["log_amount"] = (log_amt - mean) / scale
                else:
                    df_proc["log_amount"] = log_amt

            if "Time" in cols and "hour_sin" in (self.feature_names or []):
                t_vals = df_proc["Time"].astype(float).values
                hour = (t_vals / 3600.0) % 24.0
                df_proc["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
                df_proc["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

            if self.feature_names and set(self.feature_names).issubset(set(df_proc.columns)):
                return df_proc[self.feature_names]

        return df

    def predict(
        self,
        inputs: Any,
        return_numpy: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Generate class predictions or calibrated probability scores."""
        model = self._model
        model_cls = model.__class__.__name__.lower()

        # XGBoost Booster handling
        if "booster" in model_cls and ("xgb" in getattr(model.__class__, "__module__", "").lower() or hasattr(model, "inplace_predict")):
            import xgboost as xgb

            # Handle DataFrame inputs with preprocessing
            if hasattr(inputs, "columns"):
                inputs = self._preprocess_dataframe(inputs)

            if isinstance(inputs, xgb.DMatrix):
                dmat = inputs
            else:
                x_np = self._to_numpy(inputs)
                if x_np.ndim == 1:
                    x_np = x_np.reshape(1, -1)

                if self.feature_names and len(self.feature_names) == x_np.shape[1]:
                    dmat = xgb.DMatrix(x_np, feature_names=self.feature_names)
                else:
                    dmat = xgb.DMatrix(x_np)

            # Check if calibrator is present
            if self.calibrator_info and self.calibrator_info.get("method") == "sigmoid":
                a = float(self.calibrator_info.get("a", 1.0))
                b = float(self.calibrator_info.get("b", 0.0))
                margin = model.predict(dmat, output_margin=True)
                p_calibrated = 1.0 / (1.0 + np.exp(a * margin + b))
                preds = p_calibrated
            else:
                preds = model.predict(dmat)

            # Format 1D probability output into 2D [1-p, p] binary classification probabilities
            if hasattr(preds, "ndim") and preds.ndim == 1:
                if np.all((preds >= 0.0) & (preds <= 1.0)):
                    preds = np.column_stack([1.0 - preds, preds])
                else:
                    preds = preds.reshape(-1, 1)

        # Standard predict_proba interface (e.g. XGBClassifier, LGBMClassifier, CatBoostClassifier)
        elif hasattr(model, "predict_proba"):
            x_np = self._to_numpy(inputs)
            preds = model.predict_proba(x_np)
        elif hasattr(model, "predict"):
            x_np = self._to_numpy(inputs)
            preds = model.predict(x_np)
            if hasattr(preds, "ndim") and preds.ndim == 1:
                if np.all((preds >= 0.0) & (preds <= 1.0)):
                    preds = np.column_stack([1.0 - preds, preds])
                else:
                    preds = preds.reshape(-1, 1)
        else:
            raise TypeError(f"Tree model {type(model)} does not expose predict() or predict_proba().")

        preds_np = np.asarray(preds, dtype=np.float32)
        if return_numpy:
            return preds_np
        return torch.from_numpy(preds_np).float()

    def predict_proba(self, inputs: Any, **kwargs: Any) -> Union[torch.Tensor, np.ndarray]:
        """Explicit predict_proba returning class probability distribution."""
        return self.predict(inputs, return_numpy=kwargs.get("return_numpy", False), **kwargs)

    def get_logits(self, inputs: Any, **kwargs: Any) -> torch.Tensor:
        """Estimate or return raw margins/logits for scoring and metric calculations."""
        model = self._model
        model_cls = model.__class__.__name__.lower()

        if "booster" in model_cls and ("xgb" in getattr(model.__class__, "__module__", "").lower() or hasattr(model, "inplace_predict")):
            import xgboost as xgb
            x_np = self._to_numpy(inputs)
            if x_np.ndim == 1:
                x_np = x_np.reshape(1, -1)
            if self.feature_names and len(self.feature_names) == x_np.shape[1]:
                dmat = xgb.DMatrix(x_np, feature_names=self.feature_names)
            else:
                dmat = xgb.DMatrix(x_np)
            raw = model.predict(dmat, output_margin=True)
            raw_np = np.asarray(raw, dtype=np.float32)
            if raw_np.ndim == 1:
                # Format as binary logits [-margin, margin]
                raw_np = np.column_stack([-raw_np, raw_np])
            return torch.from_numpy(raw_np).float()

        preds = self.predict(inputs, return_numpy=True)
        # Safe log-odds transformation
        eps = 1e-7
        preds_clipped = np.clip(preds, eps, 1.0 - eps)
        logits = np.log(preds_clipped)
        return torch.from_numpy(logits.astype(np.float32)).float()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.predict(*args, **kwargs)
