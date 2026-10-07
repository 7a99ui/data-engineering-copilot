# Dataset d'évaluation basé sur le contenu RÉEL des documents ingérés :
# docs/guides/spark.md
# docs/guides/airflow.md
# docs/guides/pipeline_errors.md
# docs/incidents/incidents_history.md
# docs/guides/runbook_customer_etl.md
# docs/guides/kafka.md
# docs/guides/postgresql.md
# docs/guides/data_quality.md
# docs/guides/duckdb.md
# docs/sql/schema.sql

# GROUND TRUTH = résumé fidèle du contenu réel de chaque document
# RAGAS comparera les réponses du système avec ces références

EVALUATION_DATASET = [

    # ── Question 1 — Spark OutOfMemoryError ──
    # Source : docs/guides/spark.md + docs/incidents/incidents_history.md (INC-2025-002)
    {
        "question": "Why does Spark fail with OutOfMemoryError?",
        "ground_truth": (
            "Spark fails with OutOfMemoryError when the memory allocated to executors "
            "is insufficient. Solutions include increasing spark.executor.memory to 4g, "
            "increasing spark.driver.memory to 2g, reducing spark.executor.cores to 2, "
            "and enabling disk spilling with spark.memory.fraction=0.8. "
            "In incident INC-2025-002, data volume increased by 40% and executor memory "
            "was only 1g. The fix was to use a broadcast join for the small customers table "
            "and repartition transactions by customer_id before the join."
        )
    },

    # ── Question 2 — Task not serializable ──
    # Source : docs/guides/spark.md + docs/incidents/incidents_history.md (INC-2025-005)
    {
        "question": "How to fix the Spark Task not serializable error?",
        "ground_truth": (
            "The Task not serializable error occurs when a non-serializable object is used "
            "in a Spark transformation. The solution is to use local variables instead of "
            "references to class objects. In incident INC-2025-005, a UDF referenced "
            "self.config from the DataCleaner class which is not serializable. "
            "The fix was to copy needed values into local variables before defining the UDF "
            "and replace the Python UDF with native Spark functions."
        )
    },

    # ── Question 3 — Airflow task stuck in queued ──
    # Source : docs/guides/airflow.md + docs/incidents/incidents_history.md (INC-2025-006)
    {
        "question": "Why are Airflow tasks stuck in queued state?",
        "ground_truth": (
            "Airflow tasks stuck in queued state means the worker cannot pick up the task. "
            "Solutions include checking that workers are running, increasing the number of "
            "workers, and checking the broker connection to Redis or RabbitMQ. "
            "In incident INC-2025-006, the Redis broker container had stopped after a "
            "disk-full condition. The fix was to run docker system prune, restart Redis "
            "and the workers, and add a disk usage alert at 80 percent."
        )
    },

    # ── Question 4 — Kafka consumer lag ──
    # Source : docs/guides/kafka.md + docs/incidents/incidents_history.md (INC-2025-003)
    {
        "question": "What causes Kafka consumer lag and how to fix it?",
        "ground_truth": (
            "Kafka consumer lag grows when consumers are slower than producers. "
            "Solutions include adding consumers to the group up to the number of partitions, "
            "increasing max.poll.records and optimizing processing time, and increasing "
            "the number of partitions. "
            "In incident INC-2025-003, processing took longer than max.poll.interval.ms "
            "causing constant rebalances. The fix was to reduce max.poll.records from 2000 "
            "to 200, increase max.poll.interval.ms to 600000, and scale consumers from 2 to 6."
        )
    },

    # ── Question 5 — Pipeline connection refused ──
    # Source : docs/guides/pipeline_errors.md + docs/incidents/incidents_history.md (INC-2025-001)
    {
        "question": "What causes a connection refused error in a data pipeline?",
        "ground_truth": (
            "Connection refused errors are caused by the database service being stopped, "
            "incorrect host or port in configuration, or a firewall blocking the connection. "
            "Solutions include checking service status with docker compose ps, verifying "
            "environment variables, and testing with pg_isready. "
            "In incident INC-2025-001, the PostgreSQL container was killed by the OOM Killer "
            "after a heavy analytical query consumed container memory. The fix was to restart "
            "the container, increase mem_limit to 4g, reduce work_mem, and add a healthcheck."
        )
    },

    # ── Question 6 — customer_etl transform_spark failure ──
    # Source : docs/guides/runbook_customer_etl.md
    {
        "question": "What should I do when the transform_spark task fails in customer_etl?",
        "ground_truth": (
            "When transform_spark fails, likely causes are OutOfMemoryError, "
            "Task not serializable, or data skew on customer_id. "
            "Actions to take: read the Spark driver log and check the failed stage in the "
            "Spark UI, raise spark.executor.memory for the run, and refer to incidents "
            "INC-2025-002 and INC-2025-005 for previous cases. "
            "The pipeline SLA requires completion before 04:00 and typical duration is "
            "35 minutes."
        )
    },

    # ── Question 7 — Data quality checks ──
    # Source : docs/guides/data_quality.md
    {
        "question": "What data quality checks are applied in the customer_etl pipeline?",
        "ground_truth": (
            "The customer_etl pipeline applies six quality checks: "
            "not_null on customers requiring customer_id and email to be non-null (blocking), "
            "unique on customers requiring customer_id to be unique (blocking), "
            "range on transactions requiring amount between 0 and 1000000 (blocking), "
            "referential integrity checking customer_id exists in customers (warning), "
            "volume check requiring row count within 30 percent of 7-day average (warning), "
            "and freshness check requiring max created_at less than 2 hours old (blocking). "
            "Blocking failures stop the pipeline while warnings allow it to continue."
        )
    },

    # ── Question 8 — DuckDB lock error ──
    # Source : docs/guides/duckdb.md + docs/incidents/incidents_history.md (INC-2025-007)
    {
        "question": "How to fix the DuckDB Could not set lock on file error?",
        "ground_truth": (
            "The DuckDB lock error occurs when another process already holds a write "
            "connection on the database file since DuckDB allows only a single writer. "
            "Solutions include closing the other process or opening the file with "
            "read_only=True, and serializing writers through one service. "
            "In incident INC-2025-007, an analyst left a notebook connected to the warehouse "
            "in read-write mode. The fix was to close the notebook connection, enforce "
            "read_only=True for analytical access, and add a retry with delay in the load task."
        )
    },

    # ── Question 9 — PostgreSQL too many connections ──
    # Source : docs/guides/postgresql.md
    {
        "question": "How to fix the PostgreSQL too many connections error?",
        "ground_truth": (
            "The FATAL too many connections error occurs when client connections exceed "
            "max_connections. Solutions include checking active connections with "
            "SELECT count(*) FROM pg_stat_activity, using a connection pooler such as "
            "PgBouncer, closing idle connections and reducing pool size of applications. "
            "Increasing max_connections should be a last resort since each connection "
            "uses memory. Recommended settings include shared_buffers=2GB, work_mem=64MB, "
            "and max_connections=200."
        )
    },

    # ── Question 10 — Schema mismatch ──
    # Source : docs/guides/data_quality.md + docs/incidents/incidents_history.md (INC-2025-004)
    {
        "question": "What is schema drift and how to handle it in a data pipeline?",
        "ground_truth": (
            "Schema drift occurs when a source adds, removes or renames a column. "
            "Solutions include validating incoming data against an explicit schema using "
            "Pydantic, Great Expectations or Spark StructType, versioning schemas with a "
            "Schema Registry, failing fast on removed columns, tolerating new columns and "
            "logging them. "
            "In incident INC-2025-004, the source team renamed phone_number to phone "
            "without notice causing quality_checks to fail. The fix was to update the "
            "extraction mapping, add schema validation at extraction time, and set up a "
            "schema-change notification agreement."
        )
    },
]


def get_questions() -> list[str]:
    """
    Retourne uniquement les questions du dataset.
    Utilisée pour faire tourner le système RAG sur chaque question.
    """
    return [item["question"] for item in EVALUATION_DATASET]


def get_ground_truths() -> list[str]:
    """
    Retourne uniquement les réponses de référence.
    Utilisées par RAGAS pour comparer avec les réponses du système.
    """
    return [item["ground_truth"] for item in EVALUATION_DATASET]