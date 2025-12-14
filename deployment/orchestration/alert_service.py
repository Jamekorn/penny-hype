"""Alerting service stub (email/slack hooks)."""

def send_alert(message: str, level: str = "info"):
    print(f"ALERT [{level}] - {message}")
