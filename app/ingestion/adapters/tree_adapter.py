"""
Tree-based model adapter implementation for standardizing XGBoost, LightGBM, and CatBoost models.
"""

from typing import Any, Optional, Union
import numpy as np
import torch

from app.ingestion.adapters.base_adapter import AdapterCapabilities, BaseModelAdapter


class TreeModelAdapter(BaseModelAdapter):
    """
    Standardized adapter for tree-based models:
    - XGBoost (Booster, XGBClassifier)
    - LightGBM (Booster, LGBMClassifier)
    - CatBoost (CatBoostClassifier)

    Correctly flags supports_gradients=False so that gradient-based attacks (FGSM/PGD)
    are marked unsupported while black-box / score-based evaluations operate properly.
    """

    def __init__(
        self,
        model: Any,
        device: Union[str, torch.device] = "cpu",
        processor: Optional[Any] = None,
        data_domain: str = "tabular",
        framework: str = "tree",
    ):
        self._model = model
        self._device = str(device)
        self.processor = processor
        self.data_domain = data_domain
        self.framework = framework

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Tree models are non-differentiable in continuous PyTorch autograd."""
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

    def predict(
        self,
        inputs: Any,
        return_numpy: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Generate class predictions or probability scores."""
        x_np = self._to_numpy(inputs)
        model = self._model

        # Handle different tree library prediction APIs
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(x_np)
        elif hasattr(model, "predict"):
            preds = model.predict(x_np)
            # XGBoost native Booster.predict() returns probabilities for binary/multiclass
            if hasattr(preds, "ndim") and preds.ndim == 1:
                # If 1D probabilities for binary classification, format as [1-p, p]
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

    def get_logits(self, inputs: Any, **kwargs: Any) -> torch.Tensor:
        """Estimate or return raw margin/logits for scoring and metric calculations."""
        x_np = self._to_numpy(inputs)
        model = self._model
        if hasattr(model, "predict") and "output_margin" in getattr(model.predict, "__code__", object()).co_varnames:
            raw = model.predict(x_np, output_margin=True)
            return torch.from_numpy(np.asarray(raw, dtype=np.float32)).float()
        
        preds = self.predict(inputs, return_numpy=True)
        # Numerical safe log-odds for probability inputs
        eps = 1e-7
        preds_clipped = np.clip(preds, eps, 1.0 - eps)
        logits = np.log(preds_clipped)
        return torch.from_numpy(logits.astype(np.float32)).float()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.predict(*args, **kwargs)

