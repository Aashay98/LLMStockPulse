"""Error tracking integration using Sentry.

If ``sentry_sdk`` is not available, a lightweight stub is used so that the
application can run without the dependency.  This allows the rest of the
codebase to call ``init_sentry`` and ``capture_exception`` unconditionally.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised when sentry_sdk is installed
    import sentry_sdk
except Exception:  # pragma: no cover

    class _Stub:
        @staticmethod
        def init(*args, **kwargs) -> None:  # no-op
            return None

        @staticmethod
        def capture_exception(exc: Exception) -> None:  # no-op
            return None

    sentry_sdk = _Stub()  # type: ignore


def init_sentry(dsn: str | None) -> None:
    """Initialise Sentry error tracking if a DSN is provided."""

    if dsn:
        sentry_sdk.init(dsn=dsn, traces_sample_rate=1.0)


def capture_exception(exc: Exception) -> None:
    """Capture an exception via Sentry if available."""

    sentry_sdk.capture_exception(exc)


__all__ = ["init_sentry", "capture_exception"]
