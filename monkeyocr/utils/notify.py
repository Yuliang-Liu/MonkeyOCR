import requests
import logging
from monkeyocr.conf import app_settings

__all__ = ["notify_task_finished"]

def notify_task_finished(task_id: str, task_name: str):
    url = app_settings.task_notify_url
    if not url:
        logging.warning("No task_notify_url configured, skip notify.")
        return
    payload = {
        "task_id": task_id,
        "task_name": task_name
    }
    print(f"notify_task_finished: {payload} -> {url}")
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        logging.info(f"Task notify sent: {payload} -> {url}")
    except Exception as e:
        logging.error(f"Failed to notify task finished: {e}")
