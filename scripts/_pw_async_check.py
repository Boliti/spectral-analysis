"""Throwaway verification script for the background-job/polling training architecture.
Not a deliverable - deletes its own test user data when done."""
from __future__ import annotations

import sys
import tempfile
import time
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8765"


def make_classification_csv(n_per_class: int = 18, n_features: int = 80) -> str:
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    header = ["sample_id", "target"] + [f"{x:.1f}" for x in np.linspace(400, 1800, n_features)]
    rows.append(",".join(header))
    for cls in ["A", "B"]:
        base = rng.normal(5 if cls == "A" else 8, 1, size=n_features)
        for i in range(n_per_class):
            spectrum = base + rng.normal(0, 0.5, size=n_features)
            row = [f"{cls}_{i}", cls] + [f"{v:.4f}" for v in spectrum]
            rows.append(",".join(row))
    path = Path(tempfile.gettempdir()) / f"pwasync_{uuid.uuid4().hex[:8]}.csv"
    path.write_text("\n".join(rows), encoding="utf-8")
    return str(path)


def main() -> int:
    username = f"pwasync_{uuid.uuid4().hex[:8]}"
    password = "Test12345!"
    csv_path = make_classification_csv()
    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        try:
            reg = context.request.post(f"{BASE_URL}/register", form={
                "username": username, "password": password, "confirm_password": password,
            })
            print("register status:", reg.status)
            login = context.request.post(f"{BASE_URL}/login", form={"username": username, "password": password})
            print("login status:", login.status)

            page.goto(f"{BASE_URL}/analysis", wait_until="networkidle")
            print("loaded /analysis, console errors so far:", len(console_errors))
            page.evaluate("() => window.setWorkflowMode('train')")

            page.set_input_files("#standard-dataset-file", csv_path)
            page.click("#standard-dataset-import-btn")
            page.wait_for_function("activeDatasetId != null", timeout=30000)
            print("dataset imported:", page.evaluate("() => activeDatasetId"))

            page.evaluate("() => { modelTypeInput.value = 'compare_classification'; updateModelAdvancedSettings('compare_classification'); }")
            # leave "Подробный расчёт" UNCHECKED -> quick mode should be used (the new default)
            detailed_checked = page.evaluate("() => document.getElementById('detailed-calculation')?.checked")
            print("detailed-calculation checked (should be false/None):", detailed_checked)

            t0 = time.time()
            page.click("#train-btn")

            # Button must be disabled immediately (request returns run_id right away, not after
            # the whole pipeline finishes) - this is the actual bug report: previously this click
            # would hang the whole tab for the full GridSearchCV+kfold+bootstrap+permutation run.
            page.wait_for_function(
                "document.getElementById('train-btn').disabled === true", timeout=5000
            )
            t_disabled = time.time() - t0
            print(f"train-btn disabled after {t_disabled:.2f}s (should be near-instant, not after full computation)")

            # Capture a few stage labels while polling is in progress, proving the UI shows
            # progressive stage text instead of just freezing on one message.
            seen_stages = set()
            for _ in range(40):
                text = page.evaluate("() => document.getElementById('global-status-text')?.textContent || ''")
                if text:
                    seen_stages.add(text)
                if page.evaluate("() => document.getElementById('train-btn').disabled === false"):
                    break
                page.wait_for_timeout(1000)
            total_time = time.time() - t0
            print(f"training finished after {total_time:.1f}s; distinct stage texts seen: {len(seen_stages)}")
            for s in sorted(seen_stages):
                print("  stage:", s)

            page.wait_for_function("lastRenderedRunId != null", timeout=15000)
            run_id = page.evaluate("() => lastRenderedRunId")
            print("rendered run_id:", run_id)

            best_method = page.evaluate("() => lastResultPayload?.best_method ?? lastResultPayload?.run?.best_method")
            print("best_method:", best_method)
            quick_mode_in_results = page.evaluate(
                "() => { const rows = (lastResultPayload?.results?.method_results) || (lastResultPayload?.run?.results?.method_results) || []; "
                "return rows.map(r => (r.hyperparameter_search||{}).quick_mode); }"
            )
            print("quick_mode flags from saved hyperparameter_search reports:", quick_mode_in_results)

        finally:
            browser.close()
            try:
                import sqlite3

                conn = sqlite3.connect(str(Path(__file__).resolve().parent.parent / "users.db"))
                cur = conn.cursor()
                cur.execute("SELECT id FROM users WHERE username = ?", (username,))
                row = cur.fetchone()
                if row:
                    uid = row[0]
                    cur.execute("DELETE FROM saved_presets WHERE user_id = ?", (uid,))
                    cur.execute("DELETE FROM users WHERE id = ?", (uid,))
                    conn.commit()
                conn.close()
            except Exception as exc:
                print("cleanup warning:", exc)
            import shutil

            user_dir = Path(__file__).resolve().parent.parent / "user_data"
            for child in user_dir.glob(f"user_*{username}*"):
                shutil.rmtree(child, ignore_errors=True)
            Path(csv_path).unlink(missing_ok=True)

    print("console_errors:", console_errors)
    print("page_errors:", page_errors)
    return 0 if not console_errors and not page_errors else 1


if __name__ == "__main__":
    sys.exit(main())
