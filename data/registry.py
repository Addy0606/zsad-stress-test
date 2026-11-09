"""Build any dataset from its YAML config, so scripts never hard-code a dataset class or path.

    configs/datasets/<name>.yaml   what the dataset is (loader, categories, image size)
    configs/paths/<profile>.yaml   where each dataset lives on this machine (local, kaggle)

Example:
    ds = build_dataset("mvtec", paths="local")
    ds = build_dataset("visa", paths="kaggle", overrides=["categories=[candle]"])
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from data.base import AnomalyDataset
from data.medical_brain import BMADBrain
from data.medical_headct import HeadCT
from data.mvtec import MPDD, MVTecAD
from data.visa import VisA

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

LOADERS: dict[str, type[AnomalyDataset]] = {
    "mvtec": MVTecAD,
    "visa": VisA,
    "mpdd": MPDD,
    "bmad_brain": BMADBrain,
    "headct": HeadCT,
}


def dataset_names() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "datasets").glob("*.yaml"))


def load_dataset_config(name: str, paths: str = "local", overrides: list[str] | None = None) -> DictConfig:
    """Merge the dataset YAML with its root from the paths profile, then apply `key=value` overrides."""
    ds_file = CONFIG_DIR / "datasets" / f"{name}.yaml"
    paths_file = CONFIG_DIR / "paths" / f"{paths}.yaml"
    if not ds_file.exists():
        raise ValueError(f"unknown dataset '{name}'; available: {dataset_names()}")
    if not paths_file.exists():
        raise ValueError(f"unknown paths profile '{paths}': {paths_file} not found")
    cfg = OmegaConf.load(ds_file)
    roots = OmegaConf.load(paths_file)
    cfg.root = roots[name]
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def build_dataset(name: str, paths: str = "local", overrides: list[str] | None = None) -> AnomalyDataset:
    cfg = load_dataset_config(name, paths, overrides)
    loader = LOADERS[cfg.loader]
    categories = list(cfg.categories) if cfg.get("categories") else None
    return loader(root=cfg.root, categories=categories, image_size=cfg.get("image_size"))
