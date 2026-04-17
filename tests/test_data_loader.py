import io

import pytest

from src.data_loader import load_sensor_csv


def test_load_sensor_csv_rejects_empty_file():
    with pytest.raises(ValueError, match="Could not parse"):
        load_sensor_csv(io.BytesIO(b""))


def test_load_sensor_csv_rejects_single_column():
    body = b"only_one_column\nvalue1\nvalue2\n"
    with pytest.raises(ValueError, match="at least 2 columns"):
        load_sensor_csv(io.BytesIO(body))


def test_load_sensor_csv_parses_simple_two_column_csv():
    body = b"a,b\n1,2\n3,4\n"
    df = load_sensor_csv(io.BytesIO(body))
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
