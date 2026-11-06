from fastapi.testclient import TestClient

import app.main as main
from app.db.repository import get_request_log


def test_health():
    with TestClient(main.app) as client:
        body = client.get("/health").json()
    assert body["database"] == "connected"
    assert body["documents_expected"] == 6


def test_forecast_success_is_logged(monkeypatch, sample_result):
    monkeypatch.setattr(main, "run_forecast", lambda task: (sample_result, ["financial_data_extractor({})"]))
    with TestClient(main.app) as client:
        resp = client.post("/forecast", json={"task": "Forecast the next quarter for TCS"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["result"]["forecast"]["confidence"] == "medium"

        row = get_request_log(body["request_id"])
        assert row.status == "success"
        assert row.response_payload["result"]["forecast_period"] == "Q3 FY27"

        logged = client.get(f"/forecast/{body['request_id']}").json()
        assert logged["request"]["task"] == "Forecast the next quarter for TCS"


def test_forecast_error_is_logged(monkeypatch):
    def boom(task):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(main, "run_forecast", boom)
    with TestClient(main.app) as client:
        resp = client.post("/forecast", json={"task": "Forecast the next quarter for TCS"})
    assert resp.status_code == 500
    request_id = resp.json()["detail"]["request_id"]
    row = get_request_log(request_id)
    assert row.status == "error" and "LLM unavailable" in row.error_message


def test_rejects_too_short_task():
    with TestClient(main.app) as client:
        assert client.post("/forecast", json={"task": "hi"}).status_code == 422


def test_quota_errors_map_to_429(monkeypatch):
    def quota(task):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: requests per day limit exceeded")

    monkeypatch.setattr(main, "run_forecast", quota)
    with TestClient(main.app) as client:
        resp = client.post("/forecast", json={})
    assert resp.status_code == 429
    assert "quota" in resp.json()["detail"]["error"]
