# Day 1: ML Fundamentals - Question 10
## Data Infrastructure for Fraud Detection (No Code Version)

**Question:** For this fraud detection system, explain how you would design the data infrastructure. Focus on pipelines, storage strategy, and data quality.

---

## Answer:

While Q9 focused on ML serving, this question is about the data foundation. As a data engineer, this is your specialty.

---

## The Data Challenge

### Volume
- **10,000 transactions/second = 864M/day**
- **1.5KB per transaction** (raw + features)
- **Daily: 1.3 TB/day**
- **Annual: 475 TB/year**
- **7-year retention: 3.3 PB total**

### Velocity
- Real-time streaming data
- Sub-100ms feature retrieval required
- Continuous aggregate updates
- High-throughput ingestion

### Variety
- Transaction events (structured)
- User behavior logs (semi-structured)
- External API enrichment (third-party data)
- Investigation labels (delayed, sparse)

### Veracity
- Duplicate detection needed
- Missing field handling required
- Data validation critical
- Outlier identification important

---

## High-Level Data Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                              │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Transaction  │  │   External   │  │     User     │          │
│  │   Systems    │  │     APIs     │  │   Behavior   │          │
│  │              │  │              │  │     Logs     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER (Kafka)                        │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Topic: transactions-raw        (10K msg/sec)              │  │
│  │  Topic: transactions-enriched   (10K msg/sec)              │  │
│  │  Topic: fraud-labels            (50 msg/sec)               │  │
│  │  Topic: model-predictions       (10K msg/sec)              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  5 brokers, 3x replication, 7-day retention                       │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│              STREAM PROCESSING LAYER (Flink)                      │
│                                                                    │
│  Job 1: Real-time Feature Engineering     (<50ms latency)        │
│  Job 2: Aggregate Computation              (1h/24h/7d/30d)       │
│  Job 3: Data Quality Monitoring            (continuous)          │
│                                                                    │
│  Exactly-once semantics, state checkpointing, handles late data   │
└──────┬──────────────────────┬───────────────────────────────────┘
       │                      │
       ▼                      ▼
┌──────────────────┐  ┌──────────────────────────────────────────┐
│  STORAGE LAYER   │  │       BATCH PROCESSING (Spark)            │
│                  │  │                                            │
│  Tier 1: Redis   │  │  Nightly Jobs (2 AM - 6 AM):             │
│  (Hot, <10ms)    │  │  • Feature engineering (60 min)          │
│  351GB in memory │  │  • Aggregate computation (30 min)        │
│                  │  │  • Model training (60 min)               │
│  Tier 2: S3      │  │  • Model evaluation (15 min)             │
│  (Warm/Cold)     │  │  • Data quality report (10 min)          │
│  3.3 PB          │  └──────────────────────────────────────────┘
│                  │
│  Tier 3: Snowflake│
│  (Analytics)     │
│  BI & Reporting  │
└──────────────────┘
```

---

## Component 1: Data Ingestion (Kafka)

### Why Kafka?

**Advantages:**
- Handles 10K+ msg/sec easily (proven at this scale)
- Low latency (<10ms producer to broker)
- Durable (messages persisted to disk)
- Scalable (horizontal partitioning)
- Replayable (can reprocess historical data)
- Decouples producers from consumers

**Architecture:**

```
┌──────────────────────────────────────────────────────────────────┐
│                      KAFKA CLUSTER                                │
│                                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Broker 1 │  │ Broker 2 │  │ Broker 3 │  │ Broker 4 │ ...    │
│  │          │  │          │  │          │  │          │        │
│  │ Leader   │  │ Follower │  │ Leader   │  │ Follower │        │
│  │ Part 0-5 │  │ Part 0-5 │  │ Part 6-11│  │ Part 6-11│        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                    │
│  Configuration:                                                   │
│  • 5 brokers for redundancy                                      │
│  • Replication factor: 3 (can lose 2 brokers)                   │
│  • 30 partitions per topic (parallel processing)                 │
│  • 7-day retention (allows replay)                               │
└──────────────────────────────────────────────────────────────────┘
```

### Topics Design

**1. transactions-raw**
```
Purpose: All incoming transactions (unprocessed)
Rate: 10,000 msg/sec
Partitioning: By user_id (maintains ordering per user)
Retention: 7 days
Consumers: 
  • Flink (real-time processing)
  • Spark (batch processing)
  • Archival job (to S3)

Message schema (Avro):
{
  "transaction_id": "txn_123456",
  "user_id": "user_789",
  "amount": 127.50,
  "merchant_id": "merch_456",
  "timestamp": "2024-12-12T10:30:45Z",
  "location": {"lat": 40.7128, "lon": -74.0060},
  "device": "mobile",
  ...
}
```

**2. transactions-enriched**
```
Purpose: After feature engineering (ready for ML)
Rate: 10,000 msg/sec
Partitioning: By user_id
Retention: 7 days
Consumers:
  • ML prediction service
  • Data warehouse (Snowflake)
  • Monitoring systems

Message schema:
{
  "transaction_id": "txn_123456",
  "user_id": "user_789",
  "features": {
    "amount": 127.50,
    "merchant_category": "restaurant",
    "distance_from_home": 15.2,
    "transaction_count_24h": 3,
    "avg_amount_30d": 85.30,
    ...
  },
  "timestamp": "2024-12-12T10:30:45Z"
}
```

**3. fraud-labels**
```
Purpose: Investigation outcomes (ground truth)
Rate: ~50 msg/sec (sparse, only completed investigations)
Partitioning: By transaction_id
Retention: 90 days
Consumers:
  • Training pipeline
  • Performance evaluation
  • Model monitoring

Message schema:
{
  "transaction_id": "txn_123456",
  "is_fraud": true,
  "investigation_complete_time": "2024-12-18T14:22:10Z",
  "fraud_type": "stolen_card",
  "confidence": "high"
}
```

**4. model-predictions**
```
Purpose: All ML predictions (for audit and monitoring)
Rate: 10,000 msg/sec
Partitioning: By transaction_id
Retention: 30 days
Consumers:
  • Monitoring dashboards
  • Audit system
  • Model evaluation

Message schema:
{
  "transaction_id": "txn_123456",
  "fraud_probability": 0.72,
  "decision": "review",
  "model_version": "v2024-12-11",
  "features_used": {...},
  "prediction_latency_ms": 18,
  "timestamp": "2024-12-12T10:30:45Z"
}
```

### Data Quality at Ingestion

**Schema Validation (Avro):**
```
On message arrival:
  1. Validate against Avro schema
  2. Check required fields present
  3. Verify data types correct
  4. Validate value ranges
  
  If validation fails:
    → Route to dead-letter queue
    → Alert data quality team
    → Do not process
```

**Duplicate Detection:**
```
Window: 60 seconds
Key: transaction_id

For each message:
  1. Check Redis cache for transaction_id
  2. If exists within 60s window:
     → Duplicate detected
     → Log and drop
  3. If not exists:
     → Add to cache with 60s TTL
     → Process normally

Prevents:
  • Network retries from creating duplicates
  • Client-side resubmissions
```

**Completeness Checks:**
```
Required fields:
  • transaction_id (must be unique)
  • user_id (must exist in user database)
  • amount (must be positive)
  • timestamp (must be recent, within 5 minutes)
  • merchant_id (must exist in merchant database)

If missing critical field:
  → Route to dead-letter queue
  → Cannot process without core data
```

**Dead-Letter Queue (DLQ):**
```
Purpose: Store invalid messages for investigation

DLQ Topics:
  • dlq-schema-errors (schema validation failures)
  • dlq-duplicates (duplicate transactions)
  • dlq-incomplete (missing required fields)
  • dlq-processing-errors (downstream failures)

Monitoring:
  • Alert if DLQ rate >0.1%
  • Daily review of DLQ messages
  • Pattern analysis to fix upstream issues
```

---

## Component 2: Stream Processing (Flink)

### Why Flink?

**Advantages:**
- True streaming (not micro-batch like Spark Streaming)
- Low latency (<100ms processing)
- Exactly-once semantics (critical for money)
- Stateful processing (maintains aggregates)
- Event-time handling (handles late data correctly)
- Built-in windowing

**Architecture:**

```
┌──────────────────────────────────────────────────────────────────┐
│                    FLINK CLUSTER                                  │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Job Manager  │  │ Job Manager  │  │ Job Manager  │          │
│  │   (Leader)   │  │  (Standby)   │  │  (Standby)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              TASK MANAGERS (Workers)                      │   │
│  │                                                            │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │   │
│  │  │   TM 1     │  │   TM 2     │  │   TM 3     │  ...    │   │
│  │  │  (4 slots) │  │  (4 slots) │  │  (4 slots) │         │   │
│  │  └────────────┘  └────────────┘  └────────────┘         │   │
│  │                                                            │   │
│  │  State Backend: RocksDB (for large state)                │   │
│  │  Checkpointing: Every 1 minute to S3                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Processing Jobs

**Job 1: Real-Time Feature Engineering**

```
┌──────────────────────────────────────────────────────────────────┐
│         JOB 1: REAL-TIME FEATURE ENGINEERING                      │
│                                                                    │
│  Input: transactions-raw (Kafka)                                 │
│  Output: transactions-enriched (Kafka)                           │
│  Latency: <50ms P95                                              │
│                                                                    │
│  Pipeline:                                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 1. Read from Kafka                         (5ms)           │  │
│  │    └─ Consume transaction event                            │  │
│  │                                                              │  │
│  │ 2. Merchant Metadata Lookup                (10ms)          │  │
│  │    └─ Query Redis for merchant info                        │  │
│  │    └─ Merchant category, risk score, location              │  │
│  │                                                              │  │
│  │ 3. Derived Feature Calculation             (15ms)          │  │
│  │    └─ Distance from user home location                     │  │
│  │    └─ Time since last transaction                          │  │
│  │    └─ Transaction time features (hour, day of week)        │  │
│  │    └─ Amount deviation from user average                   │  │
│  │                                                              │  │
│  │ 4. Geospatial Computations                 (10ms)          │  │
│  │    └─ Country/region identification                        │  │
│  │    └─ Distance calculations                                │  │
│  │    └─ Timezone adjustments                                 │  │
│  │                                                              │  │
│  │ 5. Write to Kafka                          (5ms)           │  │
│  │    └─ Publish enriched transaction                         │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Optimizations:                                                   │
│  • Async I/O for Redis lookups (parallel)                        │
│  • Caching frequent merchant lookups                             │
│  • Pre-computed geospatial indexes                               │
│                                                                    │
│  Total Latency: 45ms P95 ✓                                       │
└──────────────────────────────────────────────────────────────────┘
```

**Job 2: Aggregate Computation**

```
┌──────────────────────────────────────────────────────────────────┐
│         JOB 2: AGGREGATE COMPUTATION                              │
│                                                                    │
│  Input: transactions-raw (Kafka)                                 │
│  Output: Redis feature store (writes)                            │
│  Windows: 1h, 24h, 7d, 30d (sliding windows)                    │
│                                                                    │
│  Aggregates Computed:                                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Per User:                                                   │  │
│  │ • transaction_count_[window]                               │  │
│  │ • total_spend_[window]                                     │  │
│  │ • avg_transaction_amount_[window]                          │  │
│  │ • max_transaction_amount_[window]                          │  │
│  │ • distinct_merchants_[window]                              │  │
│  │ • distinct_countries_[window]                              │  │
│  │ • declined_count_[window]                                  │  │
│  │ • device_change_count_[window]                             │  │
│  │                                                              │  │
│  │ Per Merchant:                                               │  │
│  │ • transaction_volume_[window]                              │  │
│  │ • unique_customers_[window]                                │  │
│  │ • fraud_rate_[window]                                      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Windowing Strategy:                                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Sliding Window:                                             │  │
│  │   Window Size: 24 hours                                    │  │
│  │   Slide Interval: 1 minute                                 │  │
│  │   Trigger: Every minute, emit updated aggregates           │  │
│  │                                                              │  │
│  │ Example for transaction_count_24h:                         │  │
│  │   10:00 → Count transactions from 10:00 yesterday to now  │  │
│  │   10:01 → Count transactions from 10:01 yesterday to now  │  │
│  │   (Continuously updated)                                    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  State Management:                                               │
│  • RocksDB state backend (handles large state)                  │
│  • Checkpointing every 1 minute to S3                           │
│  • State TTL configured (auto-expire old data)                  │
│                                                                    │
│  Write to Redis:                                                 │
│  • Batch updates (not per transaction)                          │
│  • Every 1 minute, flush all updated aggregates                 │
│  • Atomic updates using Redis pipelining                        │
└──────────────────────────────────────────────────────────────────┘
```

**Job 3: Data Quality Monitoring**

```
┌──────────────────────────────────────────────────────────────────┐
│         JOB 3: DATA QUALITY MONITORING                            │
│                                                                    │
│  Input: transactions-raw (Kafka)                                 │
│  Output: Metrics (Prometheus), Alerts                            │
│                                                                    │
│  Checks:                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 1. Volume Anomalies                                         │  │
│  │    • Track messages per minute                             │  │
│  │    • Expected: 600K/minute (10K/sec × 60)                 │  │
│  │    • Alert if: <300K or >900K (±50% threshold)            │  │
│  │                                                              │  │
│  │ 2. Distribution Shifts                                      │  │
│  │    • Track feature distributions (mean, std, percentiles)  │  │
│  │    • Compare to baseline (last 7 days)                     │  │
│  │    • Alert if: Significant shift detected (PSI > 0.25)    │  │
│  │                                                              │  │
│  │ 3. Missing Value Rates                                      │  │
│  │    • Track % missing for each field                        │  │
│  │    • Alert if: Missing rate >1% for critical fields       │  │
│  │                                                              │  │
│  │ 4. Data Freshness                                          │  │
│  │    • Track lag between event time and processing time      │  │
│  │    • Alert if: Lag >60 seconds                            │  │
│  │                                                              │  │
│  │ 5. Schema Evolution                                         │  │
│  │    • Detect new fields or changed types                    │  │
│  │    • Alert for unexpected schema changes                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Metrics Emitted (every 1 minute):                               │
│  • transaction_volume_per_minute                                 │
│  • feature_distribution_stats (per feature)                      │
│  • missing_value_percentage (per field)                          │
│  • data_lag_seconds (P50, P95, P99)                             │
│  • dlq_message_rate                                              │
└──────────────────────────────────────────────────────────────────┘
```

### State Management & Checkpointing

```
┌──────────────────────────────────────────────────────────────────┐
│              FLINK STATE MANAGEMENT                               │
│                                                                    │
│  State Backend: RocksDB                                          │
│  Why: Can handle state larger than memory (GBs to TBs)          │
│                                                                    │
│  State Stored:                                                    │
│  • Windowed aggregates (per user, per merchant)                 │
│  • Recent transaction history (for deduplication)                │
│  • Feature computation intermediate results                       │
│                                                                    │
│  Checkpointing:                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Frequency: Every 1 minute                                   │  │
│  │ Location: S3 (durable storage)                              │  │
│  │ Mode: Exactly-once                                          │  │
│  │                                                              │  │
│  │ Process:                                                     │  │
│  │ 1. Flink pauses processing (brief)                         │  │
│  │ 2. Takes snapshot of all state                             │  │
│  │ 3. Writes to S3 with version number                        │  │
│  │ 4. Resumes processing                                       │  │
│  │                                                              │  │
│  │ Recovery:                                                    │  │
│  │ • On failure, load last successful checkpoint               │  │
│  │ • Replay Kafka from checkpoint position                    │  │
│  │ • Exactly-once guarantees maintained                        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Handling Late Data:                                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Watermark: 5 minutes behind current time                   │  │
│  │ Allowed lateness: 5 minutes                                │  │
│  │                                                              │  │
│  │ Example:                                                     │  │
│  │ Current time: 10:05                                        │  │
│  │ Watermark: 10:00 (5 min behind)                           │  │
│  │ Accept events: Up to 09:55 (5 min allowed lateness)       │  │
│  │                                                              │  │
│  │ If event arrives from 09:50:                               │  │
│  │   → Within allowed lateness → Process                      │  │
│  │   → Update relevant windows                                │  │
│  │                                                              │  │
│  │ If event arrives from 09:40:                               │  │
│  │   → Too late → Drop or route to late-data topic           │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component 3: Storage Layer (Multi-Tier)

### Tier 1: Feature Store (Redis)

```
┌──────────────────────────────────────────────────────────────────┐
│                  REDIS FEATURE STORE (HOT)                        │
│                                                                    │
│  Purpose: Ultra-fast feature serving (<10ms)                     │
│                                                                    │
│  Architecture:                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │        REDIS CLUSTER (Sharded by user_id)                  │  │
│  │                                                              │  │
│  │  Shard 1        Shard 2        Shard 3        Shard 4      │  │
│  │  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    │  │
│  │  │Master 1│    │Master 2│    │Master 3│    │Master 4│    │  │
│  │  │Users   │    │Users   │    │Users   │    │Users   │    │  │
│  │  │A-F     │    │G-L     │    │M-R     │    │S-Z     │    │  │
│  │  └───┬────┘    └───┬────┘    └───┬────┘    └───┬────┘    │  │
│  │      │             │             │             │          │  │
│  │  ┌───┴────┐    ┌───┴────┐    ┌───┴────┐    ┌───┴────┐    │  │
│  │  │Replica │    │Replica │    │Replica │    │Replica │    │  │
│  │  │   1    │    │   2    │    │   3    │    │   4    │    │  │
│  │  └────────┘    └────────┘    └────────┘    └────────┘    │  │
│  │                                                              │  │
│  │  5th Node: Additional replica for failover                 │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Data Stored:                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ USER PROFILES (200 GB):                                     │  │
│  │   Key: user:{user_id}:profile                              │  │
│  │   Value (Hash):                                             │  │
│  │     - account_age_days                                      │  │
│  │     - account_type                                          │  │
│  │     - kyc_status                                            │  │
│  │     - home_location_lat, home_location_lon                 │  │
│  │     - historical_risk_score                                │  │
│  │     - avg_transaction_amount                               │  │
│  │     - total_lifetime_transactions                          │  │
│  │                                                              │  │
│  │ AGGREGATE FEATURES (150 GB):                                │  │
│  │   Key: user:{user_id}:agg:{window}                         │  │
│  │   Windows: 1h, 24h, 7d, 30d                                │  │
│  │   Value (Hash):                                             │  │
│  │     - transaction_count                                     │  │
│  │     - total_spend                                           │  │
│  │     - distinct_merchants (Set)                             │  │
│  │     - max_amount                                            │  │
│  │     - declined_count                                        │  │
│  │     - device_changes                                        │  │
│  │                                                              │  │
│  │ MERCHANT DATA (1 GB):                                       │  │
│  │   Key: merchant:{merchant_id}                              │  │
│  │   Value (Hash):                                             │  │
│  │     - name, category, location                             │  │
│  │     - risk_score                                            │  │
│  │     - fraud_rate_30d                                        │  │
│  │     - transaction_volume                                    │  │
│  │                                                              │  │
│  │ TOTAL: 351 GB in memory                                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Performance:                                                     │
│  • Latency: <10ms P95                                            │
│  • Throughput: 100K+ ops/sec per node                           │
│  • Read from replicas (load distribution)                        │
│  • Write to master (consistency)                                 │
│                                                                    │
│  Persistence:                                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ RDB Snapshots: Every 5 minutes                              │  │
│  │   → Full snapshot saved to disk                            │  │
│  │   → Used for recovery after restart                        │  │
│  │                                                              │  │
│  │ AOF (Append-Only File): Every write                        │  │
│  │   → Every write operation logged                           │  │
│  │   → Replay on startup if needed                            │  │
│  │   → Provides durability                                     │  │
│  │                                                              │  │
│  │ S3 Backups: Hourly                                          │  │
│  │   → RDB snapshot uploaded to S3                            │  │
│  │   → Cross-region replication                               │  │
│  │   → Disaster recovery                                       │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Tier 2: Data Lake (S3)

```
┌──────────────────────────────────────────────────────────────────┐
│                     DATA LAKE (S3)                                │
│                                                                    │
│  Purpose: Long-term storage, training, analytics                 │
│                                                                    │
│  Organization:                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ s3://fraud-detection-prod/                                  │  │
│  │                                                              │  │
│  │ ├── raw/                                                     │  │
│  │ │   └── transactions/                                       │  │
│  │ │       └── date=2024-12-12/                               │  │
│  │ │           └── hour=14/                                    │  │
│  │ │               ├── part-00001.parquet                     │  │
│  │ │               ├── part-00002.parquet                     │  │
│  │ │               └── ...                                     │  │
│  │ │                                                            │  │
│  │ ├── processed/                                              │  │
│  │ │   ├── features/                                           │  │
│  │ │   │   └── date=2024-12-12/                               │  │
│  │ │   │       └── user_features.parquet                      │  │
│  │ │   ├── aggregates/                                         │  │
│  │ │   │   └── date=2024-12-12/                               │  │
│  │ │   └── training_sets/                                      │  │
│  │ │       └── train_2024-12-12.parquet                       │  │
│  │ │                                                            │  │
│  │ ├── models/                                                  │  │
│  │ │   ├── production/                                         │  │
│  │ │   │   ├── v2024-12-12/                                   │  │
│  │ │   │   │   ├── model.xgb                                  │  │
│  │ │   │   │   ├── metadata.json                              │  │
│  │ │   │   │   └── metrics.json                               │  │
│  │ │   │   └── current -> v2024-12-12  (symlink)             │  │
│  │ │   └── experimental/                                       │  │
│  │ │                                                            │  │
│  │ ├── labels/                                                  │  │
│  │ │   └── fraud_investigations/                              │  │
│  │ │       └── date=2024-12-12/                               │  │
│  │ │           └── labels.parquet                             │  │
│  │ │                                                            │  │
│  │ └── analytics/                                               │  │
│  │     ├── model_performance/                                  │  │
│  │     ├── data_quality_reports/                              │  │
│  │     └── business_metrics/                                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Partitioning Strategy:                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Partition by: date (and hour for high-volume)              │  │
│  │                                                              │  │
│  │ Benefits:                                                    │  │
│  │ • Query only relevant partitions                           │  │
│  │ • Efficient for time-range queries                         │  │
│  │ • Easy to apply lifecycle policies                         │  │
│  │                                                              │  │
│  │ Example query:                                              │  │
│  │ "Get all transactions from last 7 days"                    │  │
│  │ → Scan only 7 date partitions                              │  │
│  │ → Ignore older data                                         │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  File Format: Parquet                                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Why Parquet?                                                │  │
│  │ • Columnar storage (efficient for analytics)               │  │
│  │ • High compression (5-10x vs JSON)                         │  │
│  │ • Schema evolution support                                 │  │
│  │ • Predicate pushdown (filter at storage layer)            │  │
│  │                                                              │  │
│  │ Compression Stats:                                          │  │
│  │ • JSON: 1.5 KB/transaction                                 │  │
│  │ • Parquet (Snappy): 200 bytes/transaction                  │  │
│  │ • Compression ratio: 7.5x                                  │  │
│  │ • Annual savings: ~$40K storage costs                      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Lifecycle Policies:                                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ HOT (0-90 days): S3 Standard                               │  │
│  │   • Frequent access for training                           │  │
│  │   • Cost: $0.023/GB/month                                  │  │
│  │   • Size: 78 TB × $0.023 = $1,794/month                   │  │
│  │                                                              │  │
│  │ WARM (90 days - 1 year): S3 Intelligent-Tiering           │  │
│  │   • Occasional access for audits                           │  │
│  │   • Auto-transitions to cheaper tier                       │  │
│  │   • Cost: $0.0125/GB/month                                 │  │
│  │   • Size: 237 TB × $0.0125 = $2,963/month                 │  │
│  │                                                              │  │
│  │ COLD (1-7 years): S3 Glacier Deep Archive                 │  │
│  │   • Compliance/legal requirements only                     │  │
│  │   • Retrieval: 12-48 hours (acceptable)                   │  │
│  │   • Cost: $0.00099/GB/month                                │  │
│  │   • Size: 2,850 TB × $0.00099 = $2,822/month              │  │
│  │                                                              │  │
│  │ TOTAL STORAGE COST: ~$7,579/month                          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Cross-Region Replication:                                       │
│  • Primary: us-east-1                                            │
│  • Replica: us-west-2                                            │
│  • Disaster recovery RTO: 15 minutes                             │
│  • Compliance requirement                                        │
└──────────────────────────────────────────────────────────────────┘
```

### Tier 3: Data Warehouse (Snowflake)

```
┌──────────────────────────────────────────────────────────────────┐
│                  DATA WAREHOUSE (Snowflake)                       │
│                                                                    │
│  Purpose: BI, Analytics, Ad-hoc Queries, Reporting               │
│                                                                    │
│  Schema Design (Star Schema):                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ FACT TABLES:                                                │  │
│  │                                                              │  │
│  │ transactions_fact                                           │  │
│  │   - transaction_id (PK)                                    │  │
│  │   - user_id (FK)                                           │  │
│  │   - merchant_id (FK)                                       │  │
│  │   - amount                                                  │  │
│  │   - timestamp                                               │  │
│  │   - location_id (FK)                                       │  │
│  │   - is_fraud (from labels)                                 │  │
│  │   - 864M rows/day, 25B rows/month                         │  │
│  │                                                              │  │
│  │ model_predictions_fact                                      │  │
│  │   - prediction_id (PK)                                     │  │
│  │   - transaction_id (FK)                                    │  │
│  │   - fraud_probability                                      │  │
│  │   - decision (approve/review/decline)                      │  │
│  │   - model_version                                           │  │
│  │   - prediction_latency_ms                                  │  │
│  │   - 864M rows/day                                          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ DIMENSION TABLES:                                           │  │
│  │                                                              │  │
│  │ users_dimension                                             │  │
│  │   - user_id (PK)                                           │  │
│  │   - account_type                                            │  │
│  │   - kyc_status                                              │  │
│  │   - registration_date                                       │  │
│  │   - home_location                                           │  │
│  │   - 100M rows                                               │  │
│  │                                                              │  │
│  │ merchants_dimension                                         │  │
│  │   - merchant_id (PK)                                       │  │
│  │   - name                                                    │  │
│  │   - category                                                │  │
│  │   - location                                                │  │
│  │   - risk_tier                                               │  │
│  │   - 1M rows                                                 │  │
│  │                                                              │  │
│  │ date_dimension                                              │  │
│  │   - date_id (PK)                                           │  │
│  │   - date, year, month, day                                 │  │
│  │   - day_of_week, is_weekend, is_holiday                   │  │
│  │   - 10K rows (covers many years)                          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Usage Patterns:                                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 1. Business Dashboards                                      │  │
│  │    • Daily fraud prevention metrics                        │  │
│  │    • Model performance over time                           │  │
│  │    • Regional transaction patterns                         │  │
│  │    • Merchant risk analysis                                │  │
│  │                                                              │  │
│  │ 2. Ad-hoc Analysis                                          │  │
│  │    • Investigating fraud patterns                          │  │
│  │    • Feature correlation analysis                          │  │
│  │    • Cohort analysis                                        │  │
│  │    • A/B test results                                       │  │
│  │                                                              │  │
│  │ 3. Reporting                                                 │  │
│  │    • Compliance reports                                     │  │
│  │    • Financial summaries                                    │  │
│  │    • Model audit trails                                     │  │
│  │    • Data quality metrics                                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Data Loading:                                                   │
│  • Batch load from S3 every hour                                │
│  • Snowpipe for real-time ingestion (optional)                  │
│  • Incremental updates (only new data)                          │
│                                                                    │
│  Cost Optimization:                                              │
│  • Auto-suspend warehouses after 5 min idle                     │
│  • Separate compute for different workloads                     │
│  • Result caching (24 hours)                                    │
│  • Clustering keys for large tables                             │
│  • Estimated cost: $5K-10K/month                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component 4: Batch Processing (Spark)

### Nightly Job Schedule

```
┌──────────────────────────────────────────────────────────────────┐
│              BATCH PROCESSING PIPELINE (Spark)                    │
│                                                                    │
│  Schedule: Daily, 2 AM - 6 AM (4-hour window)                    │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 2:00 AM - FEATURE ENGINEERING (90 days of data)            │  │
│  │ Duration: 60 minutes                                        │  │
│  │                                                              │  │
│  │ Input: S3 raw transactions (7-90 days ago)                 │  │
│  │ Output: S3 processed features                              │  │
│  │                                                              │  │
│  │ Steps:                                                       │  │
│  │ 1. Read transactions from S3 (Parquet)                     │  │
│  │ 2. Filter: Only transactions 7-90 days old                 │  │
│  │    Why: 7 days allows fraud labels to complete             │  │
│  │ 3. Join with user profiles, merchant data                  │  │
│  │ 4. Compute all 50 features (same as real-time)            │  │
│  │ 5. Validate feature consistency                            │  │
│  │ 6. Write to S3 processed/features/                         │  │
│  │                                                              │  │
│  │ Critical: Use same feature computation code as Flink       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 3:00 AM - AGGREGATE COMPUTATION                             │  │
│  │ Duration: 30 minutes                                        │  │
│  │                                                              │  │
│  │ Compute complex long-term aggregates:                       │  │
│  │ • 30-day behavior patterns                                  │  │
│  │ • 90-day transaction statistics                            │  │
│  │ • User baseline profiles                                    │  │
│  │ • Merchant risk scores (global)                            │  │
│  │                                                              │  │
│  │ Output:                                                      │  │
│  │ • S3: Aggregate features (for training)                    │  │
│  │ • Redis: Push updated user profiles                        │  │
│  │ • Snowflake: Load for analytics                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 4:00 AM - MODEL TRAINING                                    │  │
│  │ Duration: 60 minutes                                        │  │
│  │                                                              │  │
│  │ (Details covered in Q9)                                     │  │
│  │ • Load training data (10M transactions)                    │  │
│  │ • Handle class imbalance                                    │  │
│  │ • Train XGBoost model                                       │  │
│  │ • Evaluate on test set                                      │  │
│  │ • Save model artifacts to S3                               │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 5:00 AM - MODEL EVALUATION                                  │  │
│  │ Duration: 15 minutes                                        │  │
│  │                                                              │  │
│  │ • Compare new model vs current production                   │  │
│  │ • Validation criteria checks                               │  │
│  │ • Generate evaluation report                                │  │
│  │ • Decision: Deploy or keep current                         │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 5:15 AM - DATA QUALITY REPORT                               │  │
│  │ Duration: 10 minutes                                        │  │
│  │                                                              │  │
│  │ Generate daily report:                                      │  │
│  │ • Volume trends (past 7 days)                              │  │
│  │ • Distribution analysis (all features)                      │  │
│  │ • Drift detection (PSI calculations)                       │  │
│  │ • Anomaly identification                                    │  │
│  │ • Missing value analysis                                    │  │
│  │                                                              │  │
│  │ Output: Email report to ML team                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 5:30 AM - SNOWFLAKE DATA LOAD                              │  │
│  │ Duration: 20 minutes                                        │  │
│  │                                                              │  │
│  │ • Load yesterday's transactions to Snowflake                │  │
│  │ • Update dimension tables (users, merchants)               │  │
│  │ • Refresh materialized views                               │  │
│  │ • Update business metrics tables                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 5:50 AM - CLEANUP & ARCHIVAL                                │  │
│  │ Duration: 10 minutes                                        │  │
│  │                                                              │  │
│  │ • Archive old Kafka data to S3                             │  │
│  │ • Delete expired Redis keys (TTL)                          │  │
│  │ • Trigger S3 lifecycle transitions                         │  │
│  │ • Cleanup temp files                                        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Total Pipeline Time: 4 hours (2 AM - 6 AM)                      │
│  Slack: Before business hours start                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component 5: Data Quality Framework

### Five-Layer Quality Checks

```
┌──────────────────────────────────────────────────────────────────┐
│              DATA QUALITY FRAMEWORK (5 LAYERS)                    │
│                                                                    │
│  LAYER 1: INGESTION (Kafka)                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Checks at data entry:                                       │  │
│  │ • Schema validation (Avro)                                  │  │
│  │ • Duplicate detection (60s window)                         │  │
│  │ • Required fields present                                   │  │
│  │ • Data type validation                                      │  │
│  │                                                              │  │
│  │ Action on failure:                                          │  │
│  │ → Route to dead-letter queue                               │  │
│  │ → Alert if DLQ rate >0.1%                                  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  LAYER 2: STREAMING (Flink)                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Real-time monitoring:                                       │  │
│  │ • Distribution monitoring (per feature)                     │  │
│  │ • Completeness validation (missing %  )                     │  │
│  │ • Range checks (outliers)                                   │  │
│  │ • Consistency checks (cross-field validation)              │  │
│  │                                                              │  │
│  │ Example:                                                     │  │
│  │ IF transaction_amount > $10,000                            │  │
│  │    AND user_avg_amount < $100                              │  │
│  │    THEN flag as potential outlier                          │  │
│  │                                                              │  │
│  │ Action on anomaly:                                          │  │
│  │ → Log to monitoring system                                 │  │
│  │ → Increment anomaly counter                                │  │
│  │ → Alert if anomaly rate >5%                                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  LAYER 3: BATCH (Spark)                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Deep analysis (daily):                                      │  │
│  │ • Statistical validation                                    │  │
│  │   - Mean, median, std within expected ranges              │  │
│  │   - Distribution shape (KS test)                           │  │
│  │                                                              │  │
│  │ • Correlation analysis                                      │  │
│  │   - Feature correlations consistent                        │  │
│  │   - No unexpected relationships                            │  │
│  │                                                              │  │
│  │ • Temporal consistency                                      │  │
│  │   - Day-over-day changes reasonable                        │  │
│  │   - No sudden jumps/drops                                  │  │
│  │                                                              │  │
│  │ • Referential integrity                                     │  │
│  │   - All user_ids exist in user table                      │  │
│  │   - All merchant_ids valid                                 │  │
│  │                                                              │  │
│  │ Action on issues:                                           │  │
│  │ → Generate data quality report                             │  │
│  │ → Email to ML team                                          │  │
│  │ → Block training if critical issues                        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  LAYER 4: PRE-TRAINING                                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Before model training starts:                               │  │
│  │ • Label quality checks                                      │  │
│  │   - Fraud rate in expected range (0.3-0.7%)               │  │
│  │   - Label completeness (no missing)                        │  │
│  │   - Label consistency (review disputed cases)              │  │
│  │                                                              │  │
│  │ • Feature availability                                      │  │
│  │   - All 50 features present                                │  │
│  │   - Missing rate <1% per feature                           │  │
│  │                                                              │  │
│  │ • Train/test split validation                              │  │
│  │   - No temporal leakage                                     │  │
│  │   - No data from future in training                        │  │
│  │   - Distributions similar in train/test                    │  │
│  │                                                              │  │
│  │ Action on failure:                                          │  │
│  │ → Abort training                                            │  │
│  │ → Alert ML team                                             │  │
│  │ → Investigate data pipeline                                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  LAYER 5: POST-TRAINING                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ After model training:                                       │  │
│  │ • Model sanity checks                                       │  │
│  │   - Accuracy >90% (reasonable)                             │  │
│  │   - Not predicting all same class                          │  │
│  │   - Feature importance makes sense                         │  │
│  │                                                              │  │
│  │ • Feature importance validation                             │  │
│  │   - Top features expected (transaction amount, etc.)       │  │
│  │   - No suspicious features (data leakage indicators)       │  │
│  │                                                              │  │
│  │ • Performance thresholds                                    │  │
│  │   - Meets all validation criteria                          │  │
│  │   - Better than current production model                   │  │
│  │                                                              │  │
│  │ Action on failure:                                          │  │
│  │ → Do not deploy model                                       │  │
│  │ → Investigate training data                                │  │
│  │ → Check for pipeline issues                                │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Data Quality Dashboard

```
┌──────────────────────────────────────────────────────────────────┐
│              DATA QUALITY DASHBOARD                               │
│                                                                    │
│  SECTION 1: VOLUME METRICS                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Transactions Per Second:                                    │  │
│  │ [=========================================] 10,247 TPS ✓    │  │
│  │ Expected: 10,000 ± 1,000                                   │  │
│  │                                                              │  │
│  │ Daily Volume:                                               │  │
│  │ [=========================================] 864.2M txns ✓  │  │
│  │ Expected: 864M ± 50M                                       │  │
│  │                                                              │  │
│  │ Volume Trend (7 days):                                      │  │
│  │     │                                                        │  │
│  │ 900M│              ●                                        │  │
│  │ 850M│        ●  ●     ●                                    │  │
│  │ 800M│    ●                 ●  ●                            │  │
│  │     └────────────────────────────                          │  │
│  │      Mon Tue Wed Thu Fri Sat Sun                           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  SECTION 2: COMPLETENESS                                         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Missing Value Rates:                                        │  │
│  │                                                              │  │
│  │ transaction_id: 0.00% ✓                                    │  │
│  │ user_id:        0.02% ✓                                    │  │
│  │ amount:         0.01% ✓                                    │  │
│  │ merchant_id:    0.15% ✓                                    │  │
│  │ location:       0.35% ✓                                    │  │
│  │ device:         0.08% ✓                                    │  │
│  │                                                              │  │
│  │ All below 1% threshold ✓                                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  SECTION 3: VALIDITY                                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Out-of-Range Values:                                        │  │
│  │                                                              │  │
│  │ amount < 0:           0 occurrences ✓                      │  │
│  │ amount > $100K:       23 occurrences (0.0000027%) ✓       │  │
│  │ invalid timestamps:   0 occurrences ✓                      │  │
│  │ unknown merchants:    142 occurrences (0.000016%) ✓       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  SECTION 4: CONSISTENCY                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Cross-Field Validation:                                     │  │
│  │                                                              │  │
│  │ International flag matches location:   99.97% ✓            │  │
│  │ Merchant category valid:               99.99% ✓            │  │
│  │ Timestamp within 5 min of processing:  99.92% ✓            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  SECTION 5: FRESHNESS                                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Data Lag (Event Time → Processing Time):                   │  │
│  │                                                              │  │
│  │ P50:  12 seconds ✓                                         │  │
│  │ P95:  38 seconds ✓                                         │  │
│  │ P99:  67 seconds ✓                                         │  │
│  │                                                              │  │
│  │ All below 60s threshold ✓                                  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ALERTS (Past 24 Hours):                                         │
│  • No critical alerts ✓                                          │
│  • 2 warnings: Geographic distribution shifted 2% (acceptable)  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component 6: Compliance & Governance

### PCI-DSS Compliance

```
┌──────────────────────────────────────────────────────────────────┐
│                 PCI-DSS COMPLIANCE                                │
│                                                                    │
│  Requirement 1: Protect Cardholder Data                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Encryption at Rest:                                         │  │
│  │ • S3: SSE-KMS (256-bit AES)                                │  │
│  │ • Redis: Encryption enabled                                │  │
│  │ • Snowflake: Always encrypted                              │  │
│  │                                                              │  │
│  │ Encryption in Transit:                                      │  │
│  │ • All connections: TLS 1.2+                                │  │
│  │ • Kafka: SSL/TLS                                           │  │
│  │ • Redis: TLS enabled                                        │  │
│  │                                                              │  │
│  │ Data Masking:                                               │  │
│  │ • Card numbers: Only last 4 digits stored                  │  │
│  │ • PII: Hashed or tokenized                                 │  │
│  │ • Logs: Sensitive data redacted                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Requirement 2: Access Control                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Role-Based Access Control (RBAC):                          │  │
│  │                                                              │  │
│  │ Data Engineer:                                              │  │
│  │ • Read: All data                                           │  │
│  │ • Write: Pipeline configs                                  │  │
│  │ • No access: Production models (deploy)                   │  │
│  │                                                              │  │
│  │ ML Engineer:                                                │  │
│  │ • Read: Training data, model metrics                       │  │
│  │ • Write: Experimental models                               │  │
│  │ • Deploy: To staging only                                  │  │
│  │                                                              │  │
│  │ ML Architect (Principal):                                   │  │
│  │ • Read: All                                                │  │
│  │ • Write: All                                               │  │
│  │ • Deploy: To production (with approval)                   │  │
│  │                                                              │  │
│  │ Analyst:                                                     │  │
│  │ • Read: Aggregated data only (Snowflake)                  │  │
│  │ • No access: Raw transaction data                          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Requirement 3: Audit Logging                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ All actions logged:                                         │  │
│  │ • Data access (who, what, when)                           │  │
│  │ • Model deployments                                         │  │
│  │ • Configuration changes                                     │  │
│  │ • Permission changes                                        │  │
│  │                                                              │  │
│  │ Audit logs:                                                 │  │
│  │ • Immutable (append-only)                                  │  │
│  │ • 90-day retention                                          │  │
│  │ • Searchable in Snowflake                                  │  │
│  │ • Monitored for anomalies                                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Data Retention Policy

```
┌──────────────────────────────────────────────────────────────────┐
│                  DATA RETENTION POLICY                            │
│                                                                    │
│  TRANSACTIONS (7 years - Legal requirement):                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Year 1: S3 Standard (hot)                                   │  │
│  │ Year 2-7: S3 Glacier Deep Archive (cold)                   │  │
│  │ After 7 years: Permanently deleted                          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  MODELS (1 year - Audit trail):                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Current: S3 Standard                                        │  │
│  │ Previous versions: S3 Standard (1 year)                    │  │
│  │ After 1 year: Archived to Glacier                          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  TRAINING DATA (90 days - Recent patterns):                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ S3 Standard (90 days)                                       │  │
│  │ After 90 days: Deleted (raw still in archives)            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  LOGS (90 days - Troubleshooting):                               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ CloudWatch/Flink logs: 90 days                             │  │
│  │ After 90 days: Deleted                                      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  REDIS (Real-time only):                                         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ User profiles: No expiration                                │  │
│  │ Aggregates: TTL based on window (1h, 24h, 7d, 30d)       │  │
│  │ Backed up hourly to S3                                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### GDPR Compliance

```
┌──────────────────────────────────────────────────────────────────┐
│                     GDPR COMPLIANCE                               │
│                                                                    │
│  Principle 1: Data Minimization                                  │
│  • Only collect necessary data                                   │
│  • No excessive PII storage                                      │
│  • Delete when no longer needed                                  │
│                                                                    │
│  Principle 2: Purpose Limitation                                 │
│  • Data used only for fraud detection                            │
│  • Cannot repurpose for marketing                                │
│  • Clear documentation of usage                                  │
│                                                                    │
│  Principle 3: Right to Explanation                               │
│  • Model decisions must be explainable                           │
│  • SHAP values for each prediction                               │
│  • Audit trail of why transaction declined                       │
│                                                                    │
│  Principle 4: Right to be Forgotten                              │
│  • User can request data deletion                                │
│  • Automated deletion workflow                                   │
│  • Verify deletion across all systems                            │
│  • 30-day SLA for deletion requests                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component 7: Disaster Recovery

### RTO and RPO Targets

```
┌──────────────────────────────────────────────────────────────────┐
│                   DISASTER RECOVERY                               │
│                                                                    │
│  Targets:                                                         │
│  • RTO (Recovery Time Objective): 15 minutes                     │
│  • RPO (Recovery Point Objective): 5 minutes                     │
│                                                                    │
│  Meaning:                                                         │
│  • System restored within 15 min of failure                      │
│  • Lose at most 5 min of data                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Backup Strategy

```
┌──────────────────────────────────────────────────────────────────┐
│                      BACKUP STRATEGY                              │
│                                                                    │
│  REDIS (Feature Store):                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ RDB Snapshots:                                              │  │
│  │ • Every 5 minutes                                           │  │
│  │ • Saved locally on Redis nodes                             │  │
│  │ • Uploaded to S3 every hour                                │  │
│  │                                                              │  │
│  │ AOF (Append-Only File):                                     │  │
│  │ • Every write logged                                        │  │
│  │ • Fsync every second                                        │  │
│  │                                                              │  │
│  │ Recovery:                                                    │  │
│  │ 1. Automatic failover to replica (<5 sec)                  │  │
│  │ 2. If all nodes fail, restore from S3 snapshot             │  │
│  │ 3. Replay AOF for last changes                             │  │
│  │ 4. Total recovery time: <10 minutes                        │  │
│  │ 5. Data loss: <5 minutes (since last snapshot/AOF sync)   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  KAFKA (Message Queue):                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Replication:                                                │  │
│  │ • 3x replication factor                                     │  │
│  │ • Can lose 2 brokers without data loss                     │  │
│  │                                                              │  │
│  │ Retention:                                                   │  │
│  │ • 7 days of messages kept                                   │  │
│  │ • Can replay from any point                                │  │
│  │                                                              │  │
│  │ Recovery:                                                    │  │
│  │ • Automatic (built-in replication)                         │  │
│  │ • No manual intervention needed                            │  │
│  │ • Data loss: None (if < 2 broker failures)                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  S3 (Data Lake):                                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Built-in Durability: 99.999999999% (11 nines)             │  │
│  │                                                              │  │
│  │ Cross-Region Replication:                                   │  │
│  │ • Primary: us-east-1                                       │  │
│  │ • Replica: us-west-2                                        │  │
│  │ • Automatic replication                                     │  │
│  │                                                              │  │
│  │ Versioning:                                                  │  │
│  │ • All objects versioned                                     │  │
│  │ • Can recover deleted/overwritten files                    │  │
│  │                                                              │  │
│  │ Recovery:                                                    │  │
│  │ • Failover to replica region                               │  │
│  │ • Total time: <5 minutes (DNS update)                      │  │
│  │ • Data loss: None (continuous replication)                 │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Failure Scenarios

```
┌──────────────────────────────────────────────────────────────────┐
│              DISASTER RECOVERY SCENARIOS                          │
│                                                                    │
│  SCENARIO 1: Redis Master Node Failure                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Problem: Master node crashes                                │  │
│  │                                                              │  │
│  │ Automatic Response:                                         │  │
│  │ 1. Sentinel detects failure (< 5 seconds)                  │  │
│  │ 2. Promotes replica to master                              │  │
│  │ 3. Redirects traffic                                        │  │
│  │ 4. Spawns new replica                                       │  │
│  │                                                              │  │
│  │ User Impact: 5-10 seconds of elevated latency              │  │
│  │ Data Loss: None (replication is synchronous)               │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  SCENARIO 2: Kafka Broker Failure                                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Problem: Kafka broker crashes                               │  │
│  │                                                              │  │
│  │ Automatic Response:                                         │  │
│  │ 1. Other brokers detect failure                            │  │
│  │ 2. Partition leadership transferred                        │  │
│  │ 3. Consumers/producers reconnect                           │  │
│  │                                                              │  │
│  │ User Impact: Brief pause (< 10 seconds)                    │  │
│  │ Data Loss: None (3x replication)                           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  SCENARIO 3: Complete Regional Failure                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Problem: Entire AWS region goes down                        │  │
│  │                                                              │  │
│  │ Manual Response (DR Runbook):                               │  │
│  │ 1. Detect regional outage (monitoring)                     │  │
│  │ 2. Update DNS to failover region                           │  │
│  │ 3. Scale up services in backup region                      │  │
│  │ 4. Verify data replication                                 │  │
│  │ 5. Resume operations                                        │  │
│  │                                                              │  │
│  │ Total Time: 15-30 minutes                                   │  │
│  │ User Impact: 15-30 min downtime                            │  │
│  │ Data Loss: < 5 minutes (replication lag)                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  SCENARIO 4: Data Corruption                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Problem: Bad data in pipeline                               │  │
│  │                                                              │  │
│  │ Response:                                                    │  │
│  │ 1. Data quality checks detect corruption                   │  │
│  │ 2. Alert ML team immediately                               │  │
│  │ 3. Stop affected pipelines                                 │  │
│  │ 4. Replay from Kafka (last known good point)              │  │
│  │ 5. Fix data quality issue                                  │  │
│  │ 6. Resume processing                                        │  │
│  │                                                              │  │
│  │ Total Time: 1-4 hours (depends on issue)                   │  │
│  │ Data Loss: None (replay capability)                        │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Interview Answer Framework

**"For data infrastructure supporting 10K TPS fraud detection at scale:**

### Ingestion (Kafka)
- **5-broker cluster, 10K msg/sec**
- **4 topics:** raw, enriched, labels, predictions
- **7-day retention** for replay capability
- **Schema validation** (Avro), duplicate detection, DLQ for failures

### Stream Processing (Flink)
- **Real-time features** (<50ms latency)
- **Sliding window aggregates** (1h, 24h, 7d, 30d)
- **Exactly-once semantics** (critical for money)
- **State checkpointing** every minute to S3

### Storage (Multi-tier)
- **Redis:** 351GB in-memory, <10ms latency, 5-node cluster
- **S3:** 3.3PB tiered storage (hot/warm/cold), Parquet format (7.5x compression)
- **Snowflake:** Analytics queries, star schema, $5K-10K/month

### Batch Processing (Spark)
- **Nightly feature engineering** (90 days of data, 60 min)
- **Daily model training** (10M transactions, 60 min)
- **Data quality monitoring** and reporting

### Data Quality (5 Layers)
- **Layer 1:** Ingestion validation (schema, duplicates)
- **Layer 2:** Streaming monitoring (distributions, anomalies)
- **Layer 3:** Batch analysis (statistical validation)
- **Layer 4:** Pre-training checks (labels, features, leakage)
- **Layer 5:** Post-training validation (model sanity)

### Compliance & DR
- **PCI-DSS:** Encryption (rest/transit), RBAC, audit logging
- **GDPR:** Data minimization, right to explanation, deletion workflow
- **DR:** RTO 15 min, RPO 5 min, multi-region, automated failover

**As a Data Engineer, this is my core expertise:**
- End-to-end data quality at every stage
- 99.9% uptime with comprehensive DR
- Compliance-ready (PCI-DSS, GDPR, SOX)
- Cost-optimized ($7.5K/month storage with tiering)
- Training-serving consistency (same code, feature store)

**Key Principles:**
- Separate hot path (real-time) from cold path (batch)
- Multiple storage tiers by access pattern (Redis/S3/Snowflake)
- Quality checks at every stage (catch issues early)
- Comprehensive monitoring and alerting
- 15-minute disaster recovery capability"

---

**END OF Q10**

*This question demonstrates your deep data engineering expertise - the foundation that makes ML systems reliable and scalable.*
