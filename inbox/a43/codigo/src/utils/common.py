"""Funciones utilitarias compartidas."""

import os
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carga configuración YAML con override desde variables de entorno.

    Args:
        config_path: Ruta al archivo de configuración YAML.

    Returns:
        Diccionario con la configuración completa.
    """
    load_dotenv()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_data_path = os.getenv("DATA_PATH")
    if env_data_path:
        config["paths"]["raw_data"] = os.path.join(
            env_data_path, os.path.basename(config["paths"]["raw_data"])
        )

    env_models_path = os.getenv("MODELS_PATH")
    if env_models_path:
        config["paths"]["model_artifacts"] = env_models_path
        config["paths"]["model_checkpoint_dir"] = env_models_path

    return config


def to_numpy(tensor: Any) -> np.ndarray:
    """Convierte tensor PyTorch, lista o escalar a numpy array.

    Args:
        tensor: Tensor de PyTorch, numpy array, lista o escalar.

    Returns:
        Array de numpy.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def seed_everything(
    seed: int = 42,
    deterministic_strict: bool = False,
    deterministic_warn_only: bool = False,
) -> None:
    """Fija seeds y opciones de determinismo para reproducibilidad.

    Args:
        seed: Valor de la semilla.
        deterministic_strict: Si True, activa determinismo estricto en PyTorch.
        deterministic_warn_only: Si True, no falla ante operaciones no deterministas.
    """
    # Debe fijarse antes de operaciones CUDA/cuBLAS para máxima reproducibilidad.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if deterministic_strict:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_strict:
        torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Union[str, Path]) -> Path:
    """Crea directorio si no existe y retorna el Path.

    Args:
        path: Ruta del directorio.

    Returns:
        Objeto Path del directorio creado.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
