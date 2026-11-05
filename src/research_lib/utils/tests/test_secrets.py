"""Tests for the secrets helpers: the repo-root ``.env`` loader and
``check_auth``.

Target: src/research_lib/utils/secrets.py
Run with: pytest src/research_lib/utils/tests/test_secrets.py -v
"""

import pytest

from research_lib.utils.secrets import ALL_SERVICES, check_auth


class TestCheckAuth:
    """``check_auth`` requires the named secrets and nothing else."""

    def test_missing_selected_service_raises(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "present")
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        with pytest.raises(PermissionError, match="WANDB_API_KEY"):
            check_auth(["WANDB_API_KEY"])

    def test_only_the_selected_services_are_required(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "present")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        # W&B is present; a missing HF token must not matter.
        check_auth(["WANDB_API_KEY"])

    def test_default_is_every_service(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(PermissionError) as exc:
            check_auth()
        message = str(exc.value)
        assert all(service in message for service in ALL_SERVICES)

    def test_all_present_passes(self, monkeypatch):
        for service in ALL_SERVICES:
            monkeypatch.setenv(service, "present")
        check_auth()
        check_auth(ALL_SERVICES)
