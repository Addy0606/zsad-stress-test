"""Build a detector from configs/methods/<name>.yaml for a given target dataset.

The config can say which checkpoint to use per target (cross-dataset protocol) and which
parameters change per domain (e.g. VisualAD's top-k ratio for medical images):

    checkpoint_by_target: {mvtec: path/a.pth, default: path/b.pth}
    params_by_domain:     {medical: {topk_ratio: 0.05}}
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from methods.base import ZeroShotDetector

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "methods"


def _classes() -> dict[str, type[ZeroShotDetector]]:
    # Imported here so that importing the registry doesn't import every model library.
    from methods.baseline import CLIPPrompt, DINOMutualNN
    from methods.clip_based import AnomalyCLIP
    from methods.dino_based import AnomalyVFM
    from methods.vit_langfree import VisualAD

    return {c.__name__: c for c in (AnomalyCLIP, AnomalyVFM, VisualAD, CLIPPrompt, DINOMutualNN)}


def method_names() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def resolve_method_config(name: str, target: str, domain: str, overrides: list[str] | None = None) -> DictConfig:
    """Config with `params` fully resolved for this target dataset and domain."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown method '{name}'; available: {method_names()}")
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    params = OmegaConf.to_container(cfg.params, resolve=True) if "params" in cfg else {}
    if "params_by_domain" in cfg and domain in cfg.params_by_domain:
        params.update(OmegaConf.to_container(cfg.params_by_domain[domain], resolve=True))
    by_target = cfg.get("checkpoint_by_target")
    if by_target:
        ckpt = by_target.get(target, by_target["default"])
        params["checkpoint"] = str((REPO_ROOT / ckpt).resolve())
    cfg.resolved = {"target": target, "domain": domain, "params": params}
    return cfg


def build_method(cfg: DictConfig) -> ZeroShotDetector:
    """Instantiate from a config returned by `resolve_method_config`."""
    params = OmegaConf.to_container(cfg.resolved.params)
    return _classes()[cfg["class"]](**params)
