from src.alarms import Alarm, AlarmLog
from src.webhook import install_webhook_if_configured, webhook_dispatcher


def test_webhook_not_installed_when_env_missing(monkeypatch):
    monkeypatch.delenv("ALARM_WEBHOOK_URL", raising=False)
    log = AlarmLog(capacity=5)
    assert install_webhook_if_configured(log) is False


def test_webhook_installed_when_env_set(monkeypatch):
    monkeypatch.setenv("ALARM_WEBHOOK_URL", "http://example.invalid/webhook")
    log = AlarmLog(capacity=5)
    assert install_webhook_if_configured(log) is True


def test_webhook_dispatcher_fires_without_url(monkeypatch):
    monkeypatch.delenv("ALARM_WEBHOOK_URL", raising=False)
    alarm = Alarm(
        timestamp="2026-05-11T00:00:00Z",
        system_id="SYS-01",
        fault="Battery Fault",
        risk="Critical",
        confidence=0.9,
        recommendation="x",
    )
    webhook_dispatcher(alarm)
