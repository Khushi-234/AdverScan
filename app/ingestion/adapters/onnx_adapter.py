"""
ONNX Runtime adapter implementation for standardizing ONNX models in AdverScan.
"""

from typing import Any, Optional, Union
import numpy as np
import torch

from app.ingestion.adapters.base_adapter import AdapterCapabilities, BaseModelAdapter


class ONNXAdapter(BaseModelAdapter):
    """
    Standardized adapter for ONNX Runtime InferenceSession models.
    Exposes non-gradient prediction and logits interfaces compatible with
    AdverScan evaluation, black-box attacks, and validation.
    """

    def __init__(
        self,
        model: Any,
        device: Union[str, torch.device] = "cpu",
        processor: Optional[Any] = None,
        data_domain: str = "unknown",
    ):
        if not hasattr(model, "run") or not hasattr(model, "get_inputs"):
            raise TypeError(f"Expected onnxruntime.InferenceSession with run() and get_inputs(), got {type(model)}")

        self._model = model
        self._device = str(device)
        self.processor = processor
        self.data_domain = data_domain

    @property
    def capabilities(self) -> AdapterCapabilities:
        """ONNX Runtime sessions are non-differentiable in PyTorch autograd."""
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

    def to(self, device: Any) -> "ONNXAdapter":
        self._device = str(device)
        return self

    def eval(self) -> "ONNXAdapter":
        return self

    def train(self, mode: bool = True) -> "ONNXAdapter":
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
        input_metas = self._model.get_inputs()
        if isinstance(inputs, dict):
            feed = {
                k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
                for k, v in inputs.items()
            }
        else:
            x_np = self._to_numpy(inputs)
            input_name = input_metas[0].name if input_metas else "input"
            feed = {input_name: x_np}

        outputs = self._model.run(None, feed)
        primary = np.asarray(outputs[0])
        if return_numpy:
            return primary
        return torch.from_numpy(primary).float()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.predict(*args, **kwargs)

