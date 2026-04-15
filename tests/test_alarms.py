from src.alarms import AlarmLog


def test_alarm_log_only_keeps_high_or_critical():
    log = AlarmLog(capacity=5)
    log.add_from_prediction("Normal", "Low", 0.9, "ok")
    log.add_from_prediction("Battery Fault", "High", 0.8, "check battery")
    log.add_from_prediction("Motor Overheating", "Critical", 0.95, "stop motor")
    snap = log.snapshot()
    assert len(snap) == 2
    assert {a.fault for a in snap} == {"Battery Fault", "Motor Overheating"}


def test_alarm_log_ring_buffer():
    log = AlarmLog(capacity=3)
    for i in range(5):
        log.add_from_prediction("Motor Overheating", "Critical", 0.9, f"alarm {i}")
    assert len(log.snapshot()) == 3


def test_alarm_log_webhook_fires():
    log = AlarmLog(capacity=5)
    captured = []
    log.register_webhook(lambda a: captured.append(a.fault))
    log.add_from_prediction("Battery Fault", "High", 0.7, "x")
    assert captured == ["Battery Fault"]
