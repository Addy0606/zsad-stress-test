"""Train VisualAD on MVTec AD (auxiliary data) to get the checkpoint for testing on VisA.
The authors only released the VisA-trained checkpoint; this reproduces their MVTec -> VisA recipe
from scripts/CLIP.sh: ViT-L/14@336, 518 px, 2 epochs, batch size 8, lr 1e-3, seed 111.

Runs the official train.py unchanged (from third_party/VisualAD). Because Kaggle inputs are
read-only and train.py expects <root>/meta.json, we build a writable folder of symlinks to the
MVTec categories and generate meta.json there with the repo's own generator.

Usage (on Kaggle, after setup_third_party.py):
    python scripts/train_visualad.py --paths kaggle      # MVTec root from configs/paths/kaggle.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VISUALAD = REPO / "third_party" / "VisualAD"
TARGET = VISUALAD / "weight" / "train_on_mvtec" / "CLIP.pth"
PILLOW_FOR_TRAINING = "pillow==11.3.0"  # last release line that accepts the authors' fromarray call
MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mvtec-root", default=None, help="MVTec root (default: from --paths profile)")
    p.add_argument("--paths", default="kaggle", help="paths profile used when --mvtec-root is not given")
    p.add_argument("--work", default="/tmp/visualad_train")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--copy-to", default=None, help="also copy the checkpoint here (e.g. /kaggle/working)")
    a = p.parse_args()

    if a.mvtec_root is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from data.registry import load_dataset_config  # noqa: E402

        a.mvtec_root = str(load_dataset_config("mvtec", a.paths).root)
    if not (Path(a.mvtec_root) / "bottle" / "train").is_dir():
        sys.exit(f"MVTec root looks wrong (no bottle/train): {a.mvtec_root}")
    print("MVTec root:", a.mvtec_root, flush=True)

    work = Path(a.work)
    link_root = work / "mvtec"
    link_root.mkdir(parents=True, exist_ok=True)
    for cat in MVTEC_CATEGORIES:
        link = link_root / cat
        if not link.exists():
            os.symlink(Path(a.mvtec_root) / cat, link, target_is_directory=True)

    # meta.json with the repo's own generator (same file list the authors trained on)
    sys.path.insert(0, str(VISUALAD / "generate_dataset_json"))
    from mvtec import MVTecSolver  # noqa: E402

    MVTecSolver(root=str(link_root)).run()

    # The authors' dataset.py calls Image.fromarray(float64 array, mode="L"), which Pillow 13 rejects
    # ("Invalid mode for data type"). Give only the training process an older Pillow, so evaluation
    # keeps using the same environment as every other method.
    old_pil = work / "pillow11"
    if not (old_pil / "PIL").is_dir():
        pip = [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(old_pil), PILLOW_FOR_TRAINING]
        subprocess.run(pip, check=True)
    save = work / "checkpoints"
    env = {
        **os.environ,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # train.py enables deterministic algorithms
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # less fragmentation on a 16 GB T4
        "PYTHONPATH": os.pathsep.join([str(old_pil), os.environ.get("PYTHONPATH", "")]),
    }
    cmd = [
        sys.executable,
        "train.py",
        "--train_data_path",
        str(link_root),
        "--save_path",
        str(save),
        "--train_dataset",
        "mvtec",
        "--backbone",
        "ViT-L/14@336px",
        "--epoch",
        str(a.epochs),
        "--batch_size",
        str(a.batch_size),
        "--device",
        "cuda:0",
    ]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=VISUALAD, env=env, check=True)

    ckpt = save / f"epoch_{a.epochs}.pth"
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpt, TARGET)
    print(f"checkpoint -> {TARGET}")
    if a.copy_to:
        Path(a.copy_to).mkdir(parents=True, exist_ok=True)
        shutil.copy(ckpt, Path(a.copy_to) / "visualad_train_on_mvtec_CLIP.pth")


if __name__ == "__main__":
    main()
