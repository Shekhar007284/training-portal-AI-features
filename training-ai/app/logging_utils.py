"""Structured JSON logging with a per-request trace ID.

WHY JSON lines + contextvar?
- One JSON object per line is what Azure Monitor / ELK / Datadog parse natively.
- A contextvar carries the trace ID through every log call in the request
  without passing it around, and it is safe under async concurrency.
- The same trace ID is returned to the caller (X-Trace-ID header + response
  body) so a user can quote it in POST /feedback and we can find the request.
"""
import contextvars
import json
import logging
import sys
import time

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "trace_id": trace_id_var.get(),
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "training_ai") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})
