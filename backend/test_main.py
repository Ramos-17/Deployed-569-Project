"""Tests for the Climate LSTM FastAPI backend."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from main import app, FEATURE_COLUMNS, TARGET_COLUMN, ClimateLSTM


def _make_fake_df(n_countries: int = 2, years_per_country: int = 10) -> pd.DataFrame:
    """Return a minimal DataFrame that satisfies all backend column requirements."""
    rows = []
    base_year = 1900
    for i in range(1, n_countries + 1):
        for j in range(years_per_country):
            row = {
                "Country": f"Country_{i}",
                "country_id": str(i),
                "Year": base_year + j,
                TARGET_COLUMN: float(j) * 0.1,
            }
            for col in FEATURE_COLUMNS:
                if col != "Year":
                    row[col] = float(j + 1)
            rows.append(row)
    return pd.DataFrame(rows)


def _make_fake_assets() -> dict:
    """Build a fake assets dict with a real (tiny) model and a mock scaler."""
    df = _make_fake_df()

    model = ClimateLSTM(input_size=len(FEATURE_COLUMNS))
    model.eval()

    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: x.values.astype(float)

    return {
        "model": model,
        "scaler": scaler,
        "dataframe": df,
        "country_id_column": "country_id",
        "country_name_column": "Country",
        "year_column": "Year",
    }


@pytest.fixture()
def client():
    """TestClient with get_assets patched to avoid hitting the filesystem."""
    fake_assets = _make_fake_assets()
    with patch("main.get_assets", return_value=fake_assets):
        with TestClient(app) as c:
            yield c


class TestHealthAndRoot:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_root_returns_message(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "message" in resp.json()


class TestOptions:
    def test_returns_countries_list(self, client):
        resp = client.get("/options")
        assert resp.status_code == 200
        body = resp.json()
        assert "countries" in body
        assert len(body["countries"]) == 2

    def test_each_country_has_required_keys(self, client):
        resp = client.get("/options")
        for country in resp.json()["countries"]:
            assert "country_id" in country
            assert "country_name" in country
            assert "base_years" in country
            assert isinstance(country["base_years"], list)

    def test_base_years_require_five_consecutive_prior_rows(self, client):
        # Fake data starts at 1900 with 10 years, so the first valid target year
        # is 1905 (needs 1900-1904 as the preceding window).
        resp = client.get("/options")
        for country in resp.json()["countries"]:
            for year in country["base_years"]:
                assert year >= 1905


class TestPredict:
    VALID_PAYLOAD = {"country_id": "1", "target_year": 1905, "co2_multiplier": 1.0}

    def test_valid_request_returns_200(self, client):
        resp = client.post("/predict", json=self.VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_schema(self, client):
        resp = client.post("/predict", json=self.VALID_PAYLOAD)
        body = resp.json()
        for key in ("country_id", "country_name", "base_year", "target_year",
                    "co2_multiplier", "historical", "predicted_anomaly"):
            assert key in body

    def test_historical_has_five_points(self, client):
        resp = client.post("/predict", json=self.VALID_PAYLOAD)
        assert len(resp.json()["historical"]) == 5

    def test_historical_years_are_consecutive(self, client):
        resp = client.post("/predict", json=self.VALID_PAYLOAD)
        years = [pt["year"] for pt in resp.json()["historical"]]
        assert years == list(range(1900, 1905))

    def test_predicted_anomaly_is_float(self, client):
        resp = client.post("/predict", json=self.VALID_PAYLOAD)
        assert isinstance(resp.json()["predicted_anomaly"], float)

    def test_co2_multiplier_is_accepted(self, client):
        for multiplier in (0.5, 1.0, 1.5, 2.0):
            resp = client.post("/predict", json={**self.VALID_PAYLOAD, "co2_multiplier": multiplier})
            assert resp.status_code == 200, f"multiplier {multiplier} returned {resp.status_code}"

    def test_unknown_country_returns_404(self, client):
        resp = client.post("/predict", json={**self.VALID_PAYLOAD, "country_id": "NOPE"})
        assert resp.status_code == 404

    def test_year_without_history_returns_400(self, client):
        # Year 1901 would need 1896-1900 but data starts at 1900.
        resp = client.post("/predict", json={**self.VALID_PAYLOAD, "target_year": 1901})
        assert resp.status_code == 400

    def test_co2_multiplier_below_minimum_is_rejected(self, client):
        resp = client.post("/predict", json={**self.VALID_PAYLOAD, "co2_multiplier": 0.1})
        assert resp.status_code == 422

    def test_co2_multiplier_above_maximum_is_rejected(self, client):
        resp = client.post("/predict", json={**self.VALID_PAYLOAD, "co2_multiplier": 3.0})
        assert resp.status_code == 422

    def test_missing_country_id_is_rejected(self, client):
        resp = client.post("/predict", json={"target_year": 1905, "co2_multiplier": 1.0})
        assert resp.status_code == 422

    def test_missing_target_year_is_rejected(self, client):
        resp = client.post("/predict", json={"country_id": "1", "co2_multiplier": 1.0})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Integration tests — real model artifacts
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.skipif(
    not (
        __import__("pathlib").Path("artifacts/best_model.pth").exists()
        and __import__("pathlib").Path("artifacts/scaler.joblib").exists()
        and __import__("pathlib").Path("artifacts/processed_climate.csv").exists()
    ),
    reason="Real model artifacts not present",
)


@pytest.mark.skipif(
    not (
        __import__("pathlib").Path("artifacts/best_model.pth").exists()
        and __import__("pathlib").Path("artifacts/scaler.joblib").exists()
        and __import__("pathlib").Path("artifacts/processed_climate.csv").exists()
    ),
    reason="Real model artifacts not present",
)
class TestIntegration:
    @pytest.fixture(autouse=True)
    def real_client(self):
        # Clear lru_cache so each test class gets a fresh load.
        from main import get_assets
        get_assets.cache_clear()
        with TestClient(app) as c:
            self.client = c
        yield
        get_assets.cache_clear()

    def test_health(self):
        assert self.client.get("/health").status_code == 200

    def test_options_returns_countries(self):
        resp = self.client.get("/options")
        assert resp.status_code == 200
        assert len(resp.json()["countries"]) > 0

    def test_predict_country_1_year_1905(self):
        resp = self.client.post(
            "/predict",
            json={"country_id": "Country_1", "target_year": 1905, "co2_multiplier": 1.0},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["predicted_anomaly"], float)
        assert len(body["historical"]) == 5
