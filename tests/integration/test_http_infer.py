from fastapi.testclient import TestClient

from sat_rs_vlm.interfaces.http.app import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_http_infer() -> None:
    client = TestClient(app)
    response = client.post(
        "/infer",
        json={
            "image_path": "examples/demo_image.jpg",
            "prompt": "请检测机场跑道位置",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_type"] == "detection"
    assert payload["boxes"]
    assert "profile" in payload["raw_output"]
