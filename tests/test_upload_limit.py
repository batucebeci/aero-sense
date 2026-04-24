import io

from fastapi.testclient import TestClient

from app.main import app


def test_upload_rejects_oversized_file():
    with TestClient(app) as client:
        body = b"a,b\n" + (b"1,2\n" * (15 * 1024 * 1024))
        files = {"file": ("big.csv", io.BytesIO(body), "text/csv")}
        response = client.post("/data/upload", files=files, follow_redirects=False)
        assert response.status_code == 303

        page = client.get("/data")
        assert "too large" in page.text.lower()


def test_batch_predict_rejects_oversized_file():
    with TestClient(app) as client:
        body = b"a,b\n" + (b"1,2\n" * (15 * 1024 * 1024))
        files = {"file": ("big.csv", io.BytesIO(body), "text/csv")}
        response = client.post("/api/predict/batch", files=files)
        assert response.status_code in (400, 413)
