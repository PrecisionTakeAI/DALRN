# Database Migration Guide

## Overview

The DALRN system now supports both SQLite and PostgreSQL databases through a unified abstraction layer. This allows seamless switching between database backends without code changes.

## Database Support

### SQLite (Default)
- **Use case:** Development, testing, and small deployments
- **Configuration:** Automatic - no additional setup required
- **Database file:** `dalrn.db` (created automatically)

### PostgreSQL (Production)
- **Use case:** Production deployments, high-performance requirements
- **Configuration:** Set `DATABASE_URL` environment variable
- **Fallback:** Automatically falls back to SQLite if PostgreSQL is unavailable

## Configuration

### Environment Variables

```bash
# Use SQLite (default)
DATABASE_PATH=dalrn.db

# Use PostgreSQL
DATABASE_URL=postgresql://username:password@host:port/database_name

# Example PostgreSQL configurations
DATABASE_URL=postgresql://dalrn_user:secure_password@localhost:5432/dalrn_db
DATABASE_URL=postgresql://user@localhost/dalrn  # Local development
```

### Docker Setup for PostgreSQL

```bash
# Start PostgreSQL container
docker run -d --name dalrn-postgres \
  -e POSTGRES_DB=dalrn \
  -e POSTGRES_USER=dalrn_user \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  postgres:14

# Set environment variable
export DATABASE_URL="postgresql://dalrn_user:secure_password@localhost:5432/dalrn"
```

## Migration Process

### From SQLite to PostgreSQL

1. **Setup PostgreSQL Database:**
   ```sql
   CREATE DATABASE dalrn;
   CREATE USER dalrn_user WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE dalrn TO dalrn_user;
   ```

2. **Set Environment Variable:**
   ```bash
   export DATABASE_URL="postgresql://dalrn_user:secure_password@localhost:5432/dalrn"
   ```

3. **Restart Application:**
   The application will automatically detect PostgreSQL and create the necessary schema.

4. **Data Migration** (if needed):
   ```bash
   # Export SQLite data
   sqlite3 dalrn.db ".dump" > data_export.sql

   # Import to PostgreSQL (manual conversion may be needed)
   # Convert SQLite syntax to PostgreSQL syntax
   # Import using psql
   ```

### From PostgreSQL to SQLite

1. **Remove Environment Variable:**
   ```bash
   unset DATABASE_URL
   ```

2. **Restart Application:**
   The application will automatically use SQLite with `dalrn.db`

## Schema Comparison

Both databases use identical schema with type mappings:

| Feature | SQLite | PostgreSQL |
|---------|---------|------------|
| Primary Keys | TEXT | VARCHAR(20) |
| JSON Data | TEXT | JSONB |
| Timestamps | TIMESTAMP | TIMESTAMP |
| Text Fields | TEXT | VARCHAR(n) |

## Performance Considerations

### SQLite
- **Pros:** Zero configuration, file-based, fast for small datasets
- **Cons:** Limited concurrent writes, not suitable for high-traffic production

### PostgreSQL
- **Pros:** High performance, concurrent access, ACID compliance, JSON operations
- **Cons:** Requires separate installation and configuration

## Production Deployment

### Recommended PostgreSQL Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  dalrn-gateway:
    build: .
    environment:
      - DATABASE_URL=postgresql://dalrn_user:secure_password@postgres:5432/dalrn
    depends_on:
      - postgres

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: dalrn
      POSTGRES_USER: dalrn_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### Environment Variables for Production

```bash
# Production PostgreSQL
DATABASE_URL=postgresql://dalrn_user:secure_password@dalrn-postgres.internal:5432/dalrn

# Connection pooling (optional)
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
```

## Testing Database Functionality

### Verify Current Database Backend

```python
from services.gateway.database import get_db

db = get_db()
print(f"Database type: {type(db).__name__}")
print(f"Disputes: {db.get_dispute_count()}")
print(f"Receipts: {db.get_receipt_count()}")
```

### Test Database Operations

```bash
# Test endpoints
curl http://localhost:8001/metrics
curl http://localhost:8001/perf-test

# Submit test dispute
curl -X POST http://localhost:8001/submit-dispute \
  -H "Content-Type: application/json" \
  -d '{
    "parties": ["party1", "party2"],
    "jurisdiction": "US",
    "cid": "QmTest123456789"
  }'
```

## Troubleshooting

### PostgreSQL Connection Issues

1. **Check environment variable:**
   ```bash
   echo $DATABASE_URL
   ```

2. **Test PostgreSQL connection:**
   ```bash
   psql $DATABASE_URL -c "SELECT version();"
   ```

3. **Check logs:**
   ```bash
   tail -f gateway.log
   ```

### Common Issues

- **Port conflicts:** Ensure PostgreSQL port 5432 is available
- **Authentication:** Verify username/password in DATABASE_URL
- **Network access:** Check firewall settings for PostgreSQL
- **Database exists:** Ensure the database exists before connecting

## Backup and Recovery

### SQLite Backup
```bash
# Backup SQLite database
cp dalrn.db dalrn_backup_$(date +%Y%m%d).db

# Restore
cp dalrn_backup_20231201.db dalrn.db
```

### PostgreSQL Backup
```bash
# Backup
pg_dump $DATABASE_URL > dalrn_backup_$(date +%Y%m%d).sql

# Restore
psql $DATABASE_URL < dalrn_backup_20231201.sql
```

## Summary

The database abstraction layer provides:
- ✅ Seamless switching between SQLite and PostgreSQL
- ✅ Automatic fallback to SQLite if PostgreSQL is unavailable
- ✅ Identical API for both database backends
- ✅ Production-ready PostgreSQL support
- ✅ Zero-configuration SQLite for development

This implementation makes the DALRN system truly production-ready with enterprise-grade database support while maintaining development simplicity.