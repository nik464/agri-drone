"""
config.py - Configuration management using Pydantic.
"""
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _load_dotenv(env_file: str = ".env") -> None:
    """Read a .env file and inject its values into os.environ."""
    p = Path(env_file)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes")


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v is not None else default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None else default


class AppSettings(BaseModel):
    """Application settings from environment variables."""

    # Application
    app_name: str = "agridrone"
    app_version: str = "0.1.0"
    debug: bool = False

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("outputs/logs/agridrone.log")

    # Paths
    data_dir: Path = Path("data")
    outputs_dir: Path = Path("outputs")
    models_dir: Path = Path("models")
    configs_dir: Path = Path("configs")

    # Safety flags
    dry_run: bool = True
    safe_test_fluid_only: bool = True

    # Model
    model_name: str = "yolov8n-seg"
    model_path: Path = Path("models/yolov8n-seg.pt")
    device: str = "auto"

    # Inference
    inference_confidence: float = 0.5
    inference_iou: float = 0.45
    batch_size: int = 8

    # Geospatial
    crs_epsg: str = "EPSG:4326"
    grid_cell_size_m: float = 10.0

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False

    # Prescription
    prescription_severity_threshold: float = 0.6

    # Simulation
    sim_random_seed: int = 42

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Return settings as dictionary."""
        return {k: (str(v) if isinstance(v, Path) else v)
                for k, v in super().model_dump(**kwargs).items()}

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "AppSettings":
        """Build settings by reading a .env file + os.environ."""
        _load_dotenv(env_file)
        return cls(
            app_name=_env("APP_NAME", "agridrone"),
            debug=_env_bool("DEBUG", False),
            log_level=_env("LOG_LEVEL", "INFO"),
            log_file=Path(_env("LOG_FILE", "outputs/logs/agridrone.log")),
            data_dir=Path(_env("DATA_DIR", "data")),
            outputs_dir=Path(_env("OUTPUT_DIR", "outputs")),
            models_dir=Path(_env("MODEL_DIR", "models")),
            device=_env("DEVICE", "auto"),
            dry_run=_env_bool("DRY_RUN", True),
            safe_test_fluid_only=_env_bool("SAFE_TEST_FLUID_ONLY", True),
            inference_confidence=_env_float("INFERENCE_CONFIDENCE", 0.5),
            api_host=_env("API_HOST", "0.0.0.0"),
            api_port=_env_int("API_PORT", 8000),
        )


class ConfigManager:
    """
    Centralized configuration management.

    Handles:
    - Environment variables (.env file)
    - YAML configuration files (optional)
    - Runtime parameter overrides
    """

    def __init__(self, config_dir: str = "configs", env_file: str = ".env"):
        self.config_dir = Path(config_dir)
        self.env_file = Path(env_file)

        # Load environment settings
        self.env_settings = AppSettings.from_env(str(self.env_file))

        # Load YAML configs (best-effort, no hard dependency)
        self.yaml_configs: dict[str, dict] = {}
        self._load_yaml_configs()

    def _load_yaml_configs(self) -> None:
        """Load all YAML configuration files from config directory."""
        if not self.config_dir.exists():
            return
        try:
            import yaml as _yaml  # optional
        except ImportError:
            try:
                from omegaconf import OmegaConf
                for yf in sorted(self.config_dir.glob("*.yaml")):
                    try:
                        self.yaml_configs[yf.stem] = OmegaConf.to_container(
                            OmegaConf.load(str(yf)), resolve=True
                        )
                    except Exception:
                        pass
            except ImportError:
                pass
            return

        for yaml_file in sorted(self.config_dir.glob("*.yaml")):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    self.yaml_configs[yaml_file.stem] = _yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load config {yaml_file}: {e}")

    def get(self, key: str, default=None):
        """Get configuration value by dot-notation key."""
        parts = key.split(".")

        for config in self.yaml_configs.values():
            current = config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    break
            else:
                return current

        return getattr(self.env_settings, key.replace(".", "_"), default)

    def get_section(self, section: str) -> dict:
        """Get entire configuration section as dict."""
        return dict(self.yaml_configs.get(section, {}))

    def get_env(self) -> AppSettings:
        """Get environment settings object."""
        return self.env_settings

    def ensure_paths(self) -> None:
        """Create necessary output directories."""
        for path in [self.env_settings.outputs_dir,
                     self.env_settings.data_dir]:
            path.mkdir(parents=True, exist_ok=True)


# Global config instance
_config_manager: ConfigManager | None = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def init_config(config_dir: str = "configs", env_file: str = ".env") -> ConfigManager:
    """Initialize global configuration manager."""
    global _config_manager
    _config_manager = ConfigManager(config_dir=config_dir, env_file=env_file)
    return _config_manager
