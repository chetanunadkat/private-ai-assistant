-- Create database
CREATE DATABASE customer_data;

\c customer_data

-- Create schema
CREATE SCHEMA IF NOT EXISTS public;

-- Create tables
CREATE TABLE conversation_history (
    session_id VARCHAR(255) PRIMARY KEY,
    conversation_hist JSONB NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    response_feedback FLOAT
);

-- Remove user_id from session_info if exists
ALTER TABLE session_info DROP COLUMN IF EXISTS user_id;

-- Create read-only user after tables exist
CREATE USER "postgres_readonly" WITH PASSWORD 'readonly_password';

-- Grant permissions
GRANT CONNECT ON DATABASE customer_data TO "postgres_readonly";
GRANT USAGE ON SCHEMA public TO "postgres_readonly";
GRANT SELECT ON ALL TABLES IN SCHEMA public TO "postgres_readonly";
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO "postgres_readonly";
