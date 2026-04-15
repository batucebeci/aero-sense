from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.ai4i import load_ai4i


@pytest.fixture
def ai4i_fixture(tmp_path):
    rows = [
        "UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure,TWF,HDF,PWF,OSF,RNF",
        "1,M001,M,298.1,308.6,1551,42.8,0,0,0,0,0,0,0",
        "2,M002,M,300.0,313.0,1520,40.0,15,1,0,1,0,0,0",
        "3,M003,M,302.5,316.5,1480,52.0,150,1,0,0,1,0,0",
        "4,M004,M,299.0,310.0,1490,68.0,180,1,0,0,0,1,0",
        "5,M005,M,300.5,311.5,1500,32.0,200,1,1,0,0,0,0",
    ]
    path = tmp_path / "ai4i2020.csv"
    path.write_text("\n".join(rows), encoding="utf-8")
    return tmp_path


def test_ai4i_adapter_loads(ai4i_fixture):
    df = load_ai4i(file_name="ai4i2020.csv", base_dir=ai4i_fixture)
    assert "fault_type" in df.columns
    assert "motor_temperature" in df.columns
    assert df["system_id"].str.startswith("AI4I-TYPE-").all()
    assert df["system_id"].nunique() <= 3
    assert set(df["fault_type"]).issubset(
        {"Normal", "Motor Overheating", "Power Instability", "Vibration Fault", "Sensor Drift", "Communication Fault"}
    )


def test_ai4i_fault_label_mapping(ai4i_fixture):
    df = load_ai4i(file_name="ai4i2020.csv", base_dir=ai4i_fixture)
    assert df.iloc[0]["fault_type"] == "Normal"
    assert df.iloc[1]["fault_type"] == "Motor Overheating"
    assert df.iloc[2]["fault_type"] == "Power Instability"
    assert df.iloc[3]["fault_type"] == "Vibration Fault"
    assert df.iloc[4]["fault_type"] == "Sensor Drift"


def test_ai4i_motor_temperature_is_celsius(ai4i_fixture):
    df = load_ai4i(file_name="ai4i2020.csv", base_dir=ai4i_fixture)
    assert df["motor_temperature"].between(20, 60).all()
