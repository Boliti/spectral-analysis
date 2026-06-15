from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import app


def authenticate_test_user(client: TestClient) -> str:
    """Create a throwaway user and keep its session cookie in TestClient."""
    username = f"analysis_check_{os.getpid()}_{int(time.time() * 1000)}"
    password = "analysis-check-password"

    register_response = client.post(
        "/register",
        data={
            "username": username,
            "password": password,
            "confirm_password": password,
        },
        follow_redirects=False,
    )
    assert register_response.status_code == 303, register_response.text

    login_response = client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=False,
    )
    assert login_response.status_code == 303, login_response.text

    analysis_response = client.get("/analysis", follow_redirects=False)
    assert analysis_response.status_code == 200, analysis_response.text
    return username


def main() -> None:
    client = TestClient(app)
    username = authenticate_test_user(client)
    content = "sample_id,target,source_sheet,source_file,400,410,420,430,440,450,460,470\n"
    for idx in range(12):
        group = "A" if idx < 6 else "B"
        offset = 0 if group == "A" else 1
        values = [1 + offset, 2 + offset, 3 + offset, 2 + offset, 1.5 + offset, 1.2 + offset, 1.0 + offset, 0.8 + offset]
        content += f"s{idx},{group},Sheet1,synthetic.csv," + ",".join(map(str, values)) + "\n"

    response = client.post(
        "/analysis/dataset/import-standard",
        files={"file": ("correct.csv", content.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    dataset_id = payload["dataset_id"]
    summary = payload["summary"]
    assert summary["n_samples"] == 12
    assert "snr" in summary

    config_response = client.post("/analysis/configs", json={"blocks": {"baseline": True}, "preprocessing": {"preset": "none"}})
    assert config_response.status_code == 200, config_response.text

    compare_response = client.post(
        "/analysis/model/train",
        json={
            "dataset_id": dataset_id,
            "model_type": "compare_classification",
            "model_name": "feature_check",
            "validation": {"enabled": False, "random_state": 42},
        },
    )
    assert compare_response.status_code == 200, compare_response.text
    compare_payload = compare_response.json()
    run_id = compare_payload["run_id"]
    assert compare_payload["result"]["method_results"]

    run_response = client.get(f"/analysis/experiments/{run_id}")
    assert run_response.status_code == 200, run_response.text
    assert run_response.json()["run"]["run_id"] == run_id

    report_response = client.get(f"/analysis/reports/from-run/{run_id}?format=json")
    assert report_response.status_code == 200, report_response.text

    print(
        json.dumps(
            {
                "username": username,
                "dataset_id": dataset_id,
                "run_id": run_id,
                "snr": summary["snr"],
                "methods": [row["method"] for row in compare_payload["result"]["method_results"]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
