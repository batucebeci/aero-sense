from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_home_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Aero-Sense" in response.text


def test_pipeline_generate_train_predict(client):
    r1 = client.post("/api/dataset/generate", params={"samples_per_class": 80})
    assert r1.status_code == 200
    assert r1.json()["rows"] == 80 * 9

    r2 = client.post("/api/train")
    assert r2.status_code == 200
    assert "best_model" in r2.json()
    assert r2.json()["metrics"]["f1_macro"] > 0.6

    r3 = client.post("/api/predict")
    assert r3.status_code == 200
    assert r3.json()["rows"] > 0


@pytest.mark.parametrize(
    "path",
    [
        "/data",
        "/features",
        "/compare",
        "/train",
        "/tuning",
        "/predict",
        "/anomaly",
        "/rul",
        "/explain",
        "/counterfactual",
        "/drift",
        "/live",
        "/alarms",
        "/runs",
        "/report",
        "/docs",
        "/metrics",
    ],
)
def test_all_pages_return_200(client, path):
    response = client.get(path)
    assert response.status_code == 200


def test_metrics_endpoint_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"fastapi_requests_total" in response.content


def test_pdf_report_after_predict(client):
    client.post("/api/dataset/generate", params={"samples_per_class": 60})
    client.post("/api/train")
    client.post("/api/predict")
    response = client.get("/report/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content[:4] == b"%PDF"
