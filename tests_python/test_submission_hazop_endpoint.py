from fastapi.testclient import TestClient
from saga.api import app


def test_digital_twin_direct_hazop_endpoint_is_available():
    with TestClient(app) as client:
        response = client.post("/api/digital-twin/hazop/direct", json={
            "stationId": "submission-test", "scenarioId": "*",
            "readings": [], "equipment": [], "hazopRules": [],
            "interpret": True, "generateSop": True})
    assert response.status_code == 200, response.text
    assert "status" in response.json()
