from saga.config import Settings


def test_reads_service_hub_environment(monkeypatch):
    monkeypatch.setenv("OPEN_AI_SERVICE_HUB_API_KEY", "test-key")
    monkeypatch.setenv("SAGA_SERVICE_HUB_BASE_URL", "https://hub.example/v1")
    monkeypatch.setenv("SAGA_MODEL", "gpt-oss-120b")
    monkeypatch.setenv("SAGA_FAST_MODEL", "gpt-oss-20b")
    monkeypatch.setenv("SAGA_VISION_MODEL", "qwen2.5-vl-72b")

    settings = Settings(_env_file=None)

    assert settings.service_hub_api_key == "test-key"
    assert settings.service_hub_base_url == "https://hub.example/v1"
    assert settings.service_hub_model == "gpt-oss-120b"
    assert settings.service_hub_fast_model == "gpt-oss-20b"
    assert settings.service_hub_vision_model == "qwen2.5-vl-72b"
