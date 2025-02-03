-- Create tables
CREATE TABLE IF NOT EXISTS conversation_history (
    session_id VARCHAR(255) PRIMARY KEY,
    conversation_hist JSONB,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_summary (
    session_id VARCHAR(255) PRIMARY KEY,
    summary TEXT,
    sentiment VARCHAR(50),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    feedback FLOAT
);

-- Create a new read-only user
CREATE USER "postgres_readonly" WITH PASSWORD 'readonly_password';

-- Grant CONNECT permission to the database
GRANT CONNECT ON DATABASE customer_data TO "postgres_readonly";

-- Grant USAGE on the public schema
GRANT USAGE ON SCHEMA public TO "postgres_readonly";

-- Grant SELECT permissions on all tables in the public schema
GRANT SELECT ON ALL TABLES IN SCHEMA public TO "postgres_readonly";

-- Ensure future tables are accessible with SELECT permission
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO "postgres_readonly";
