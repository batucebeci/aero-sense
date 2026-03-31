from __future__ import annotations

import json
import os
import threading
import urllib.request

import structlog

from .alarms import Alarm

_log = structlog.get_logger("sensor_fusion.webhook")


def webhook_url() -> str | None:
    return os.environ.get("ALARM_WEBHOOK_URL")


def post_alarm(url: str, alarm: Alarm, timeout: float = 3.0) -> tuple[int | None, str | None]:
    payload = {
        "text": (
            f":rotating_light: {alarm.fault} ({alarm.risk}) on "
            f"{alarm.system_id or 'unknown system'}"
        ),
        "alarm": alarm.as_dict(),
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, None
    except Exception as exc:
        return None, str(exc)


def webhook_dispatcher(alarm: Alarm) -> None:
    url = webhook_url()
    if not url:
        return

    def _run() -> None:
        status, error = post_alarm(url, alarm)
        if error:
            _log.warning("alarm_webhook_failed", error=error, fault=alarm.fault)
        else:
            _log.info("alarm_webhook_sent", status=status, fault=alarm.fault)

    threading.Thread(target=_run, daemon=True).start()


def install_webhook_if_configured(alarm_log) -> bool:
    if webhook_url():
        alarm_log.register_webhook(webhook_dispatcher)
        return True
    return False
