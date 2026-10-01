from saga import config


def test_current_windows_user_groq_key_replaces_stale_process_copy(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "old-process-key")
    monkeypatch.setattr(
        config, "windows_user_environment",
        lambda name: "current-user-key" if name == "GROQ_API_KEY" else "",
    )
    assert config.Settings().groq_api_key == "current-user-key"
