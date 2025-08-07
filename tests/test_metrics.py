import uuid

from metrics import ERROR_COUNTER, REQUEST_COUNTER, REQUEST_LATENCY, get_metric_value


def test_metrics_record_counts_and_latency():
    user = str(uuid.uuid4())

    # Counters start at zero for a new user label
    assert get_metric_value(REQUEST_COUNTER, user=user) == 0

    REQUEST_COUNTER.labels(user=user).inc()
    assert get_metric_value(REQUEST_COUNTER, user=user) == 1

    ERROR_COUNTER.labels(user=user).inc(2)
    assert get_metric_value(ERROR_COUNTER, user=user) == 2

    with REQUEST_LATENCY.labels(user=user).time():
        pass
    assert get_metric_value(REQUEST_LATENCY, user=user) >= 0
