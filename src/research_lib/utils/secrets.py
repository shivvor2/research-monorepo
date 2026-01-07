import os
from pathlib import Path
from warnings import warn

from dotenv import load_dotenv


def _load_secrets() -> None:
    """
    Internal function to force load .env from the project root.
    The root is determined by the presence of 'pyproject.toml'.
    """
    current_path = Path(__file__).resolve()
    project_root = None

    # Traverse up the directory tree to find the anchor file
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists():
            project_root = parent
            break

    if project_root is None:
        raise FileNotFoundError(
            "Could not determine project root: 'pyproject.toml' not found in any parent directory."
        )

    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        # In a strict environment, you should log a warning here if the .env is critical.
        warn(f"Could not find .env file at {env_path}")
        pass


_load_secrets()


def check_auth():
    """
    Public function to verify auth exists.
    Call this inside your main() if you want to fail fast.
    """
    missing = []
    if not os.getenv("WANDB_API_KEY"):
        missing.append("WANDB_API_KEY")
    if not os.getenv("HF_TOKEN"):
        missing.append("HF_TOKEN")

    if missing:
        raise PermissionError(
            f"Missing secrets: {', '.join(missing)}. "
            "Ensure they are in your .env file or system environment."
        )
