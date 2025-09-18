-- DALRN Production Database Setup Script
-- Creates production database, user, and grants appropriate permissions

-- Create production database
CREATE DATABASE IF NOT EXISTS dalrn_production
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

-- Create production user with secure password
CREATE USER IF NOT EXISTS dalrn_user WITH
    PASSWORD 'DALRN_Pr0d_2024!SecureP@ss'
    NOSUPERUSER
    CREATEDB
    NOCREATEROLE
    INHERIT
    NOREPLICATION
    CONNECTION LIMIT 100;

-- Grant permissions on production database
GRANT CONNECT ON DATABASE dalrn_production TO dalrn_user;
GRANT ALL PRIVILEGES ON DATABASE dalrn_production TO dalrn_user;

-- Connect to the production database
\c dalrn_production;

-- Create schema for better organization
CREATE SCHEMA IF NOT EXISTS dalrn_schema AUTHORIZATION dalrn_user;

-- Grant schema permissions
GRANT ALL ON SCHEMA dalrn_schema TO dalrn_user;
GRANT ALL ON ALL TABLES IN SCHEMA dalrn_schema TO dalrn_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA dalrn_schema TO dalrn_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA dalrn_schema TO dalrn_user;

-- Set default search path
ALTER USER dalrn_user SET search_path TO dalrn_schema, public;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Performance settings for production
ALTER DATABASE dalrn_production SET shared_buffers = '256MB';
ALTER DATABASE dalrn_production SET effective_cache_size = '1GB';
ALTER DATABASE dalrn_production SET maintenance_work_mem = '64MB';
ALTER DATABASE dalrn_production SET checkpoint_completion_target = 0.9;
ALTER DATABASE dalrn_production SET wal_buffers = '7864kB';
ALTER DATABASE dalrn_production SET default_statistics_target = 100;
ALTER DATABASE dalrn_production SET random_page_cost = 1.1;
ALTER DATABASE dalrn_production SET effective_io_concurrency = 200;
ALTER DATABASE dalrn_production SET max_parallel_workers_per_gather = 2;
ALTER DATABASE dalrn_production SET max_parallel_workers = 8;

-- Create read-only user for reporting
CREATE USER IF NOT EXISTS dalrn_readonly WITH
    PASSWORD 'DALRN_ReadOnly_2024'
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    INHERIT
    NOREPLICATION
    CONNECTION LIMIT 10;

GRANT CONNECT ON DATABASE dalrn_production TO dalrn_readonly;
GRANT USAGE ON SCHEMA dalrn_schema TO dalrn_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA dalrn_schema TO dalrn_readonly;

-- Alter default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA dalrn_schema
    GRANT SELECT ON TABLES TO dalrn_readonly;

ALTER DEFAULT PRIVILEGES IN SCHEMA dalrn_schema
    GRANT ALL ON TABLES TO dalrn_user;

-- Output confirmation
SELECT 'Production database created successfully' AS status;
SELECT datname, pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
WHERE datname = 'dalrn_production';

SELECT usename, usecreatedb, usesuper, userepl, valuntil
FROM pg_user
WHERE usename IN ('dalrn_user', 'dalrn_readonly');