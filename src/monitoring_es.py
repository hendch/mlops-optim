import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from elasticsearch import Elasticsearch


def push_metrics_to_elasticsearch(
    *,
    experiment: str,
    metrics: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Push metrics to Elasticsearch (non-blocking by default).
    - Set STRICT_MONITORING=1 to fail if ES is unreachable.
    """
    elastic_url = os.getenv("ELASTIC_URL", "http://localhost:9200")
    index_name = os.getenv("ELASTIC_INDEX", "mlflow-metrics")
    strict = os.getenv("STRICT_MONITORING", "0") == "1"

    doc = {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "metrics": metrics,
        "params": params or {},
        "tags": tags or {},
    }

    try:
        es = Elasticsearch(elastic_url)
        es.index(index=index_name, document=doc)
    except Exception as exc:
        if strict:
            raise
        print(f"[monitoring] Elasticsearch not reachable ({elastic_url}): {exc}")
