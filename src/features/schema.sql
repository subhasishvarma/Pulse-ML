-- Feature store schema. Two layers: an entity table (static attributes),
-- and one feature table per feature group, each keyed on
-- (entity_id, feature_timestamp) so we can support point-in-time-correct joins.

CREATE TABLE IF NOT EXISTS entities (
    sk_id_curr BIGINT PRIMARY KEY,
    target SMALLINT,
    application_date DATE
);

CREATE TABLE IF NOT EXISTS features_application (
    sk_id_curr BIGINT,
    feature_timestamp TIMESTAMP,
    amt_income_total DOUBLE,
    amt_credit DOUBLE,
    amt_annuity DOUBLE,
    days_employed INTEGER,
    credit_income_ratio DOUBLE,
    annuity_income_ratio DOUBLE,
    PRIMARY KEY (sk_id_curr, feature_timestamp)
);

CREATE TABLE IF NOT EXISTS features_bureau_history (
    sk_id_curr BIGINT,
    feature_timestamp TIMESTAMP,
    num_prior_loans INTEGER,
    num_active_loans INTEGER,
    avg_days_credit_overdue DOUBLE,
    total_debt_to_credit_ratio DOUBLE,
    PRIMARY KEY (sk_id_curr, feature_timestamp)
);

CREATE TABLE IF NOT EXISTS features_previous_loans (
    sk_id_curr BIGINT,
    feature_timestamp TIMESTAMP,
    num_previous_applications INTEGER,
    num_approved INTEGER,
    num_refused INTEGER,
    approval_rate DOUBLE,
    avg_amt_application DOUBLE,
    PRIMARY KEY (sk_id_curr, feature_timestamp)
);
