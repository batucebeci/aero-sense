from __future__ import annotations

import numpy as np

from src.datasets.cmapss import CMAPSS_COLUMNS, load_cmapss


def _write_fixture(path):
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(1, 4):
        for cycle in range(1, 60):
            row = [unit, cycle] + list(rng.normal(size=len(CMAPSS_COLUMNS) - 2))
            rows.append(" ".join(str(v) for v in row))
    path.write_text("\n".join(rows), encoding="utf-8")


def test_cmapss_adapter_loads(tmp_path):
    fixture = tmp_path / "train_FD001.txt"
    _write_fixture(fixture)
    df = load_cmapss(file_name="train_FD001.txt", base_dir=tmp_path)
    assert "fault_type" in df.columns
    assert "motor_temperature" in df.columns
    assert df["system_id"].str.startswith("CMAPSS-").all()
    last_per_unit = df.groupby("system_id").tail(1)["fault_type"].unique()
    first_per_unit = df.groupby("system_id").head(1)["fault_type"].unique()
    assert set(last_per_unit) == {"Motor Overheating"}
    assert set(first_per_unit) == {"Normal"}
