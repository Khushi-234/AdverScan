"""
Query-based Black-Box Adversarial Attack for Tabular Models (XGBoost, LightGBM, CatBoost, Scikit-learn, ONNX, MLP).

Uses score/probability queries (`model.predict`) without requiring backpropagation gradients,
making it compatible with non-differentiable tree-based and classical ML models.
"""

from typing import Any, Optional
import torch

from app.attack_engine.attacks.base_attack import BaseAttack
from app.attack_engine.attack_registry import register_attack
from app.attack_engine.config import AttackConfig
from app.attack_engine.exceptions import AttackConfigurationError, AttackExecutionError
from app.attack_engine.models import AttackMetadata
from app.attack_engine.attacks.tabular._tabular_utils import (
    parse_tabular_metadata,
    validate_tabular_inputs,
)


class TabularBlackBoxAttack(BaseAttack):
    """
    Gradient-free, query-based L_inf / L_2 black-box attack for tabular models.
    Explores bounded random-search and coordinate-wise perturbations via model queries.
    """

    metadata = AttackMetadata(
        name="tabular_blackbox",
        domain="tabular",
        category="black_box",
        attack_type="black_box",
        requires_gradient=False,
        requires_loss=False,
        requires_model_weights=False,
        iterative=True,
        targeted_supported=True,
        untargeted_supported=True,
        query_based=True,
        supported_norms=["Linf", "L2"],
        computational_cost="low",
        supported_model_types=[
            "pytorch", "sklearn", "xgboost", "lightgbm", "catboost", "onnx", "huggingface", "tree"
        ],
    )

    def __init__(self, model: Any, config: Optional[AttackConfig] = None, **kwargs: Any):
        super().__init__(model=model, config=config, **kwargs)
        self.norm = str(getattr(self.config, "norm", "Linf") or "Linf")
        self.epsilon = float(getattr(self.config, "epsilon", 0.3) or 0.3)
        self.max_iter = int(
            (getattr(self.config, "params", None) or {}).get("max_iter")
            or (getattr(self.config, "params", None) or {}).get("num_steps")
            or kwargs.get("max_iter", 40)
        )
        self.targeted = bool(getattr(self.config, "targeted", False))

    def _query_scores(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.predict(x)
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out, dtype=torch.float32)
        if out.ndim == 1:
            out = torch.stack([1.0 - out, out], dim=-1)
        return out.float()

    def generate(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        targeted: Optional[bool] = None,
        target_labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        validate_tabular_inputs(inputs, labels)

        is_targeted = self.targeted if targeted is None else bool(targeted)
        eff_labels = target_labels if (is_targeted and target_labels is not None) else labels
        if eff_labels is None:
            with torch.no_grad():
                eff_labels = self._query_scores(inputs).argmax(dim=-1)
        elif not isinstance(eff_labels, torch.Tensor):
            eff_labels = torch.as_tensor(eff_labels, dtype=torch.long)

        # Parse tabular immutability / bounds if metadata is attached
        meta_info = parse_tabular_metadata(self.model, inputs.shape[-1])
        mutable_mask = meta_info.get("mutable_mask")
        if mutable_mask is None:
            mutable_mask = torch.ones(inputs.shape[-1], dtype=torch.bool)
        else:
            mutable_mask = mutable_mask.bool()
        mask_f = mutable_mask.float().unsqueeze(0)

        x_orig = inputs.detach().clone().float()
        x_adv = x_orig.clone()
        batch_size, n_features = x_orig.shape

        if self.epsilon <= 0 or not mutable_mask.any():
            return x_adv

        with torch.no_grad():
            init_scores = self._query_scores(x_orig)
            best_scores = init_scores.gather(1, eff_labels.view(-1, 1)).squeeze(1)
            # For untargeted: minimize score of true label; for targeted: maximize score of target label
            best_obj = best_scores.clone() if is_targeted else -best_scores.clone()

            rng = torch.Generator()
            rng.manual_seed(42)

            for step in range(max(1, self.max_iter)):
                # Mix dense random directions with sparse coordinate probes
                if step % 2 == 0:
                    delta = torch.empty_like(x_orig).uniform_(-1.0, 1.0, generator=rng)
                else:
                    delta = torch.zeros_like(x_orig)
                    feat_idx = torch.randint(0, n_features, (batch_size,), generator=rng)
                    signs = torch.where(torch.rand(batch_size, generator=rng) > 0.5, 1.0, -1.0)
                    delta[torch.arange(batch_size), feat_idx] = signs

                delta = delta * mask_f
                if self.norm.lower() == "l2":
                    norms = delta.view(batch_size, -1).norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
                    step_scale = self.epsilon * (0.25 + 0.75 * ((step + 1) / max(1, self.max_iter)))
                    cand = x_orig + delta / norms * step_scale
                else:
                    cand = x_orig + delta.clamp(-1.0, 1.0) * self.epsilon

                # Enforce min/max bounds if available
                min_b = meta_info.get("min_bounds")
                max_b = meta_info.get("max_bounds")
                if min_b is not None and max_b is not None:
                    cand = torch.max(torch.min(cand, max_b.unsqueeze(0)), min_b.unsqueeze(0))

                cand_scores = self._query_scores(cand)
                target_conf = cand_scores.gather(1, eff_labels.view(-1, 1)).squeeze(1)
                cand_obj = target_conf if is_targeted else -target_conf

                improved = cand_obj > best_obj
                if improved.any():
                    x_adv[improved] = cand[improved]
                    best_obj[improved] = cand_obj[improved]

        return x_adv


register_attack("tabular_blackbox", TabularBlackBoxAttack)
register_attack("blackbox_tabular", TabularBlackBoxAttack)
register_attack("tree_blackbox", TabularBlackBoxAttack)