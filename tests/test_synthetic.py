from src.synthetic_sensor_generator import FAULT_CLASSES, generate_sensor_data


def test_generates_all_classes():
    df = generate_sensor_data(samples_per_class=30, seed=1)
    assert set(df["fault_type"].unique()) == set(FAULT_CLASSES)
    assert len(df) == 30 * len(FAULT_CLASSES)


def test_battery_fault_has_lower_voltage_than_normal():
    df = generate_sensor_data(samples_per_class=80, seed=2)
    normal = df[df["fault_type"] == "Normal"]["voltage"].mean()
    battery = df[df["fault_type"] == "Battery Fault"]["voltage"].mean()
    assert battery < normal


def test_motor_overheating_has_higher_temperature():
    df = generate_sensor_data(samples_per_class=80, seed=3)
    normal_temp = df[df["fault_type"] == "Normal"]["motor_temperature"].mean()
    over_temp = df[df["fault_type"] == "Motor Overheating"]["motor_temperature"].mean()
    assert over_temp > normal_temp + 10


def test_seed_is_deterministic():
    a = generate_sensor_data(samples_per_class=20, seed=42)
    b = generate_sensor_data(samples_per_class=20, seed=42)
    assert a.equals(b)
