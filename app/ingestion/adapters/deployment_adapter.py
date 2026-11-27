"""
Generic runtime deployment adapter implementation for OpenVINO, TensorRT, Core ML, and TFLite.
"""

from typing import Any, Optional, Union
import numpy as np
import torch

from app.ingestion.adapters.base_adapter import AdapterCapabilities, BaseModelAdapter


class DeploymentRuntimeAdapter(BaseModelAdapter):
    """
    Standardized adapter for compiled / deployment runtimes:
    - OpenVINO (.xml + .bin)
    - TensorRT (.engine, .plan)
    - Core ML (.mlmodel, .mlpackage)
    - TensorFlow Lite (.tflite)

    Accurately exposes non-gradient inference capabilities (supports_gradients=False).
    """

    def __init__(
        self,
        model: Any,
        device: Union[str, torch.device] = "cpu",
        processor: Optional[Any] = None,
        data_domain: str = "unknown",
        runtime_name: str = "deployment",
    ):
        self._model = model
        self._device = str(device)
        self.processor = processor
        self.data_domain = data_domain
        self.runtime_name = runtime_name

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Deployment runtimes are non-differentiable compiled engines."""
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

    def to(self, device: Any) -> "DeploymentRuntimeAdapter":
        self._device = str(device)
        return self

    def eval(self) -> "DeploymentRuntimeAdapter":
        return self

    def train(self, mode: bool = True) -> "DeploymentRuntimeAdapter":
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
        x_np = self._to_numpy(inputs)
        model = self._model

        # TFLite Interpreter
        if hasattr(model, "get_input_details") and hasattr(model, "get_output_details"):
            in_details = model.get_input_details()
            out_details = model.get_output_details()
            model.set_tensor(in_details[0]["index"], x_np)
            model.invoke()
            out = model.get_tensor(out_details[0]["index"])
        # OpenVINO CompiledModel
        elif callable(model) or hasattr(model, "infer_new_request"):
            out = model([x_np])
            if isinstance(out, dict):
                out = next(iter(out.values()))
            elif isinstance(out, (list, tuple)) and len(out) > 0:
                out = out[0]
        # CoreML MLModel
        elif hasattr(model, "predict"):
            out = model.predict(inputs if isinstance(inputs, dict) else {"input": x_np})
            if isinstance(out, dict):
                out = next(iter(out.values()))
        else:
            raise TypeError(f"Unknown deployment model type {type(model)}.")

        out_np = np.asarray(out, dtype=np.float32)
        if return_numpy:
            return out_np
        return torch.from_numpy(out_np).float()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.predict(*args, **kwargs)

