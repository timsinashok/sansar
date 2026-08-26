"""Config loading: default.yaml < local.yaml < CLI overrides."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(CONFIG_DIR / "default.yaml")
    local = CONFIG_DIR / "local.yaml"
    if local.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(local))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Snapshot a resolved config next to a run's outputs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
