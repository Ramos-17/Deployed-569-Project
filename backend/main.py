from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"
DEFAULT_SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
DEFAULT_DATA_PATH = ARTIFACTS_DIR / "processed_climate.csv"
SEQUENCE_LENGTH = 5


FEATURE_COLUMNS = [
    "Year",
    "CO2_Emissions",
    "Population",
    "Forest_Area",
    "GDP",
    "Renewable_Energy_Usage",
    "Methane_Emissions",
    "Sea_Level_Rise",
    "Arctic_Ice_Extent",
    "Urbanization",
    "Deforestation_Rate",
    "Extreme_Weather_Events",
    "Average_Rainfall",
    "Solar_Energy_Potential",
    "Waste_Management",
    "Per_Capita_Emissions",
    "Industrial_Activity",
    "Air_Pollution_Index",
    "Biodiversity_Index",
    "Ocean_Acidification",
    "Fossil_Fuel_Usage",
    "Energy_Consumption_Per_Capita",
    "Policy_Score",
    "Average_Temperature",
]

TARGET_COLUMN = "Temperature_Anomaly"
COUNTRY_ID_CANDIDATES = ["country_id", "Country_ID", "CountryId", "Country"]
COUNTRY_NAME_CANDIDATES = ["country_name", "Country_Name", "country"]
YEAR_CANDIDATES = ["year", "Year"]


class PredictionRequest(BaseModel):
    country_id: str = Field(..., min_length=1)
    target_year: int = Field(..., ge=1900, le=2500)
    co2_multiplier: float = Field(1.0, ge=0.5, le=2.0)


class HistoricalPoint(BaseModel):
    year: int
    anomaly: float


class PredictionResponse(BaseModel):
    country_id: str
    country_name: str | None
    base_year: int
    target_year: int
    co2_multiplier: float
    historical: list[HistoricalPoint]
    predicted_anomaly: float


class ClimateLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        x = self.dropout1(lstm_out[:, -1, :])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return self.fc2(x)


app = FastAPI(title="Climate LSTM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_path(env_key: str, fallback: Path) -> Path:
    configured = os.getenv(env_key)
    return Path(configured).expanduser().resolve() if configured else fallback


def _first_matching_column(columns: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise RuntimeError(f"Missing one of the required columns: {', '.join(candidates)}")


def _normalize_country_id(value: Any) -> str:
    if pd.isna(value):
        raise RuntimeError("Encountered a row with an empty country identifier.")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _coerce_prediction(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if hasattr(value, "shape"):
        flat = value.reshape(-1)
        return float(flat[0])
    return float(value)


def _load_model(model_path: Path, input_size: int) -> nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        checkpoint.eval()
        return checkpoint

    model = ClimateLSTM(input_size=input_size)

    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if state_dict is None and isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict is None:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_assets() -> dict[str, Any]:
    model_path = _resolve_path("CLIMATE_MODEL_PATH", DEFAULT_MODEL_PATH)
    scaler_path = _resolve_path("CLIMATE_SCALER_PATH", DEFAULT_SCALER_PATH)
    data_path = _resolve_path("CLIMATE_DATA_PATH", DEFAULT_DATA_PATH)

    missing = [str(path) for path in [model_path, scaler_path, data_path] if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing required model assets. Place the files in backend/artifacts or set "
            f"CLIMATE_MODEL_PATH, CLIMATE_SCALER_PATH, and CLIMATE_DATA_PATH. Missing: {missing}"
        )

    historical_df = pd.read_csv(data_path)
    columns = historical_df.columns.tolist()
    country_id_column = _first_matching_column(columns, COUNTRY_ID_CANDIDATES)
    year_column = _first_matching_column(columns, YEAR_CANDIDATES)

    if TARGET_COLUMN not in historical_df.columns:
        raise RuntimeError(f"Missing target column '{TARGET_COLUMN}' in dataset.")

    missing_features = [column for column in FEATURE_COLUMNS if column not in historical_df.columns]
    if missing_features:
        raise RuntimeError(
            "The dataset is missing required feature columns. Update FEATURE_COLUMNS to match "
            f"the trained model. Missing: {missing_features}"
        )

    country_name_column = next(
        (candidate for candidate in COUNTRY_NAME_CANDIDATES if candidate in historical_df.columns),
        None,
    )

    working_df = historical_df.copy()
    working_df[country_id_column] = working_df[country_id_column].apply(_normalize_country_id)
    working_df[year_column] = working_df[year_column].astype(int)
    working_df = working_df.sort_values([country_id_column, year_column]).reset_index(drop=True)

    scaler = joblib.load(scaler_path)
    model = _load_model(model_path, input_size=len(FEATURE_COLUMNS))

    return {
        "model": model,
        "scaler": scaler,
        "dataframe": working_df,
        "country_id_column": country_id_column,
        "country_name_column": country_name_column,
        "year_column": year_column,
    }


def _build_sequence(df: pd.DataFrame, year_column: str, target_year: int) -> pd.DataFrame:
    sequence_years = list(range(target_year - SEQUENCE_LENGTH, target_year))
    window = df[df[year_column].isin(sequence_years)].copy()

    if len(window) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {SEQUENCE_LENGTH} historical rows before {target_year}, "
                f"but found {len(window)}."
            ),
        )

    window = window.sort_values(year_column).reset_index(drop=True)
    if window[year_column].tolist() != sequence_years:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Incomplete historical run for target year {target_year}. "
                f"Expected years: {sequence_years}."
            ),
        )

    return window


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/options")
def get_options() -> dict[str, Any]:
    assets = get_assets()
    dataframe = assets["dataframe"]
    country_id_column = assets["country_id_column"]
    country_name_column = assets["country_name_column"]
    year_column = assets["year_column"]

    options: list[dict[str, Any]] = []
    for country_id, group in dataframe.groupby(country_id_column):
        ordered = group.sort_values(year_column).reset_index(drop=True)
        year_set = set(ordered[year_column].astype(int).tolist())
        base_years = [
            year
            for year in sorted(year_set)
            if all(required_year in year_set for required_year in range(year - SEQUENCE_LENGTH, year))
        ]
        if not base_years:
            continue

        country_name = None
        if country_name_column:
            first_name = ordered[country_name_column].dropna()
            country_name = str(first_name.iloc[0]) if not first_name.empty else None

        options.append(
            {
                "country_id": country_id,
                "country_name": country_name,
                "base_years": base_years,
            }
        )

    options.sort(key=lambda item: item["country_name"] or item["country_id"])
    return {"countries": options}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    assets = get_assets()
    dataframe = assets["dataframe"]
    country_id_column = assets["country_id_column"]
    country_name_column = assets["country_name_column"]
    year_column = assets["year_column"]
    scaler = assets["scaler"]
    model = assets["model"]

    country_key = _normalize_country_id(payload.country_id)
    country_df = dataframe[dataframe[country_id_column] == country_key].copy()
    if country_df.empty:
        raise HTTPException(status_code=404, detail=f"Country '{country_key}' was not found.")

    sequence_df = _build_sequence(country_df, year_column, payload.target_year)
    sequence_features = sequence_df[FEATURE_COLUMNS].copy()

    co2_feature = "CO2_Emissions"
    if co2_feature not in sequence_features.columns:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Configured CO2 feature '{co2_feature}' is missing. Update FEATURE_COLUMNS to "
                "match your teammate's notebook."
            ),
        )

    sequence_features.loc[sequence_features.index[-1], co2_feature] *= payload.co2_multiplier

    scaled_sequence = scaler.transform(sequence_features)
    input_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0)

    with torch.inference_mode():
        prediction = model(input_tensor)

    historical = [
        HistoricalPoint(year=int(row[year_column]), anomaly=float(row[TARGET_COLUMN]))
        for _, row in sequence_df.iterrows()
    ]

    country_name = None
    if country_name_column:
        first_name = country_df[country_name_column].dropna()
        country_name = str(first_name.iloc[0]) if not first_name.empty else None

    return PredictionResponse(
        country_id=country_key,
        country_name=country_name,
        base_year=payload.target_year - 1,
        target_year=payload.target_year,
        co2_multiplier=payload.co2_multiplier,
        historical=historical,
        predicted_anomaly=_coerce_prediction(prediction),
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Climate LSTM API is running."}
