"""Paths and label order used by the published checkpoints."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = ROOT / "train_yolo" / "best6.pt"
CNN_WEIGHTS = ROOT / "train_cnn" / "best_convnext_research5.pth"

# torchvision.datasets.ImageFolder sorts directory names alphabetically.
CLASS_NAMES = (
    "battery", "computer mouse", "keyboard", "laptop", "mobile", "pcb",
    "phone charger",
)


def check_weights(path):
    """Fail clearly when weights are missing or only an LFS pointer is present."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Weights not found: {path}. Run 'git lfs pull'.")
    with path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"{path} is a Git LFS pointer. Run 'git lfs pull'.")
    return path
