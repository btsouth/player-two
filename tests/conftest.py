import pytest


@pytest.fixture(autouse=True)
def _no_real_game_notes(tmp_path, monkeypatch):
    """Tests never read or write the player's own notes about their games."""
    from playertwo import notes
    monkeypatch.setattr(notes, "CONFIG_DIR", tmp_path / "config")
