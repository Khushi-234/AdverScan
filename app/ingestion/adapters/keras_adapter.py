"""
TensorFlow / Keras adapter implementation for standardizing TF/Keras models in AdverScan.
"""

from typing import Any, Optional, Union
import numpy as np
import torch

from app.ingestion.adapters.base_adapter import AdapterCapabilities, BaseModelAdapter


class KerasAdapter(BaseModelAdapter):
    """
    Standardized adapter for TensorFlow / Keras models.
    Exposes unified prediction and logits interfaces compatible with AdverScan
    evaluation and validation.
    """

    def __init__(
        self,
        model: Any,
        device: Union[str, torch.device] = "cpu",
        processor: Optional[Any] = None,
        data_domain: str = "unknown",
    ):
        if not (hasattr(model, "predict") or callable(model)):
            raise TypeError(f"Expected callable TensorFlow/Keras model with predict(), got {type(model)}")

        self._model = model
        self._device = str(device)
        self.processor = processor
        self.data_domain = data_domain
        self._supports_gradients: Optional[bool] = None

    def _probe_supports_gradients(self) -> bool:
        """Probe whether tf.GradientTape can compute input gradients through this model."""
        if self._supports_gradients is not None:
            return self._supports_gradients
        try:
            import tensorflow as tf
            if "TFSMLayer" in type(self._model).__name__:
                self._supports_gradients = False
                return False

            in_shape = getattr(self._model, "input_shape", None) or getattr(self._model, "_savedmodel_input_shape", None)
            if isinstance(in_shape, list) and in_shape:
                in_shape = in_shape[0]
            if isinstance(in_shape, (tuple, list)) and len(in_shape) >= 2:
                dims = [1 if d is None else int(d) for d in in_shape]
            else:
                dims = [1, 4]
            x_probe = tf.Variable(tf.zeros(dims, dtype=tf.float32))
            with tf.GradientTape() as tape:
                y = self._model(x_probe, training=False)
                if isinstance(y, dict):
                    y = next(iter(y.values()))
                loss = tf.reduce_sum(tf.cast(y, tf.float32))
            grad = tape.gradient(loss, x_probe)
            self._supports_gradients = grad is not None
        except Exception:
            self._supports_gradients = False
        return self._supports_gradients

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Determine gradient capability via real tf.GradientTape probe."""
        return AdapterCapabilities(
            supports_gradients=self._probe_supports_gradients(),
            supports_predict=True,
            supports_batch_predict=True,
            supports_logits=True,
        )

    @property
    def device(self) -> str:
        return self._device

    def get_model(self) -> Any:
        return self._model

    def to(self, device: Any) -> "KerasAdapter":
        self._device = str(device)
        return self

    def eval(self) -> "KerasAdapter":
        return self

    def train(self, mode: bool = True) -> "KerasAdapter":
        return self

    def _to_numpy(self, inputs: Any) -> Any:
        if isinstance(inputs, torch.Tensor):
            return inputs.detach().cpu().numpy()
        if isinstance(inputs, dict):
            return {
                k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
                for k, v in inputs.items()
            }
        if hasattr(inputs, "values"):
            return np.asarray(inputs.values)
        return np.asarray(inputs)

    def predict(
        self,
        inputs: Any,
        return_numpy: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, np.ndarray]:
        x_in = self._to_numpy(inputs)
        if hasattr(self._model, "predict"):
            out = self._model.predict(x_in, verbose=0) if "verbose" not in kwargs else self._model.predict(x_in, **kwargs)
        else:
            out = self._model(x_in, training=False)
        if isinstance(out, dict):
            for key in ("logits", "predictions", "output_0"):
                if key in out:
                    out = out[key]
                    break
            else:
                out = next(iter(out.values()))
        if hasattr(out, "numpy"):
            out = out.numpy()
        out_np = np.asarray(out)
        if return_numpy:
            return out_np
        return torch.from_numpy(out_np).float()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.predict(*args, **kwargs)

