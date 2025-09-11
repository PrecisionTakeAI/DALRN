# DALRN Docker Orchestration Makefile
# Enforces PoDP compliance and epsilon-ledger budget constraints

.PHONY: help run-all stop-all logs test-integration clean build rebuild status health-check

# Variables
COMPOSE_FILE := infra/docker-compose.yml
COMPOSE_CMD := docker-compose -f $(COMPOSE_FILE)
ENV_FILE := infra/.env
SERVICES := gateway search fhe negotiation fl agents

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Default target
help:
	@echo "$(GREEN)DALRN Docker Orchestration Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Core Commands:$(NC)"
	@echo "  make run-all          - Start all DALRN services with PoDP validation"
	@echo "  make stop-all         - Stop all running services"
	@echo "  make restart-all      - Restart all services"
	@echo "  make status           - Show status of all services"
	@echo "  make health-check     - Verify health and PoDP compliance of all services"
	@echo ""
	@echo "$(YELLOW)Service Management:$(NC)"
	@echo "  make run-<service>    - Start specific service (gateway/search/fhe/negotiation/fl/agents)"
	@echo "  make stop-<service>   - Stop specific service"
	@echo "  make restart-<service>- Restart specific service"
	@echo "  make logs-<service>   - View logs for specific service"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make build            - Build all Docker images"
	@echo "  make rebuild          - Force rebuild all Docker images"
	@echo "  make logs             - View aggregated logs from all services"
	@echo "  make logs-follow      - Follow aggregated logs from all services"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test-integration - Run integration tests with PoDP validation"
	@echo "  make test-podp        - Validate PoDP compliance across all services"
	@echo "  make test-epsilon     - Verify epsilon budget constraints"
	@echo ""
	@echo "$(YELLOW)Monitoring:$(NC)"
	@echo "  make monitoring-up    - Start Prometheus and Grafana"
	@echo "  make monitoring-down  - Stop monitoring services"
	@echo "  make metrics          - Display current metrics"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(NC)"
	@echo "  make clean            - Remove all containers, volumes, and networks"
	@echo "  make clean-volumes    - Remove only volumes (preserves images)"
	@echo "  make clean-logs       - Clear all service logs"
	@echo "  make backup           - Backup persistent data"
	@echo "  make restore          - Restore from backup"
	@echo ""
	@echo "$(YELLOW)Configuration:$(NC)"
	@echo "  make env-setup        - Create .env file from .env.example"
	@echo "  make validate-config  - Validate configuration and PoDP settings"

# Environment setup
env-setup:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "$(GREEN)Creating .env file from .env.example...$(NC)"; \
		cp infra/.env.example $(ENV_FILE); \
		echo "$(YELLOW)Please update $(ENV_FILE) with your configuration$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists$(NC)"; \
	fi

# Validate configuration
validate-config:
	@echo "$(GREEN)Validating DALRN configuration...$(NC)"
	@echo "Checking PoDP settings..."
	@grep -q "PODP_ENABLED=true" $(ENV_FILE) && echo "✓ PoDP enabled" || echo "✗ PoDP disabled"
	@echo "Checking epsilon budget settings..."
	@grep -q "EPSILON_OVERFLOW_PREVENTION=true" $(ENV_FILE) && echo "✓ Epsilon overflow prevention enabled" || echo "✗ Epsilon overflow prevention disabled"
	@echo "Validating docker-compose configuration..."
	@$(COMPOSE_CMD) config > /dev/null && echo "✓ Docker compose configuration valid" || echo "✗ Docker compose configuration invalid"

# Build all images
build: env-setup
	@echo "$(GREEN)Building all DALRN service images...$(NC)"
	$(COMPOSE_CMD) build --parallel

# Force rebuild all images
rebuild: env-setup
	@echo "$(GREEN)Force rebuilding all DALRN service images...$(NC)"
	$(COMPOSE_CMD) build --no-cache --parallel

# Start all services
run-all: env-setup validate-config
	@echo "$(GREEN)Starting all DALRN services with PoDP validation...$(NC)"
	$(COMPOSE_CMD) up -d
	@echo "$(GREEN)Waiting for services to be healthy...$(NC)"
	@sleep 10
	@make health-check

# Stop all services
stop-all:
	@echo "$(YELLOW)Stopping all DALRN services...$(NC)"
	$(COMPOSE_CMD) down

# Restart all services
restart-all: stop-all run-all

# Show service status
status:
	@echo "$(GREEN)DALRN Service Status:$(NC)"
	@echo "===================="
	@$(COMPOSE_CMD) ps

# Health check with PoDP validation
health-check:
	@echo "$(GREEN)Performing health checks with PoDP validation...$(NC)"
	@for service in $(SERVICES); do \
		echo -n "Checking $$service... "; \
		if curl -f -s http://localhost:$$(grep "$${service^^}_PORT" $(ENV_FILE) | cut -d= -f2)/health > /dev/null 2>&1; then \
			echo "$(GREEN)✓ Healthy$(NC)"; \
		else \
			echo "$(RED)✗ Unhealthy$(NC)"; \
		fi; \
	done

# View aggregated logs
logs:
	$(COMPOSE_CMD) logs --tail=100

# Follow aggregated logs
logs-follow:
	$(COMPOSE_CMD) logs -f

# Service-specific targets
define SERVICE_TARGETS
run-$(1):
	@echo "$$(GREEN)Starting $(1) service...$$(NC)"
	$$(COMPOSE_CMD) up -d $(1)

stop-$(1):
	@echo "$$(YELLOW)Stopping $(1) service...$$(NC)"
	$$(COMPOSE_CMD) stop $(1)

restart-$(1): stop-$(1) run-$(1)

logs-$(1):
	$$(COMPOSE_CMD) logs --tail=100 $(1)

logs-follow-$(1):
	$$(COMPOSE_CMD) logs -f $(1)
endef

$(foreach service,$(SERVICES),$(eval $(call SERVICE_TARGETS,$(service))))

# Monitoring targets
monitoring-up:
	@echo "$(GREEN)Starting monitoring services (Prometheus & Grafana)...$(NC)"
	$(COMPOSE_CMD) up -d prometheus grafana
	@echo "$(GREEN)Grafana available at: http://localhost:3000$(NC)"
	@echo "$(GREEN)Prometheus available at: http://localhost:9090$(NC)"

monitoring-down:
	@echo "$(YELLOW)Stopping monitoring services...$(NC)"
	$(COMPOSE_CMD) stop prometheus grafana

metrics:
	@echo "$(GREEN)Current DALRN Metrics:$(NC)"
	@echo "===================="
	@for service in $(SERVICES); do \
		echo "\n$$service metrics:"; \
		curl -s http://localhost:$$(grep "$${service^^}_PORT" $(ENV_FILE) | cut -d= -f2)/metrics | grep -E "^dalrn_|^podp_|^epsilon_" | head -5; \
	done

# Testing targets
test-integration: env-setup
	@echo "$(GREEN)Running DALRN integration tests with PoDP validation...$(NC)"
	@echo "Building test environment..."
	$(COMPOSE_CMD) up -d
	@sleep 15
	@echo "Running PoDP compliance tests..."
	@python -m pytest tests/integration/test_podp_compliance.py -v
	@echo "Running epsilon budget tests..."
	@python -m pytest tests/integration/test_epsilon_budget.py -v
	@echo "Running service integration tests..."
	@python -m pytest tests/integration/test_services.py -v

test-podp:
	@echo "$(GREEN)Validating PoDP compliance across all services...$(NC)"
	@for service in $(SERVICES); do \
		echo "Testing $$service PoDP compliance..."; \
		curl -X POST http://localhost:$$(grep "$${service^^}_PORT" $(ENV_FILE) | cut -d= -f2)/validate-podp \
			-H "Content-Type: application/json" \
			-d '{"check_receipts": true, "validate_chain": true}' | jq .; \
	done

test-epsilon:
	@echo "$(GREEN)Verifying epsilon budget constraints...$(NC)"
	@for service in $(SERVICES); do \
		echo "Checking $$service epsilon budget..."; \
		curl -s http://localhost:$$(grep "$${service^^}_PORT" $(ENV_FILE) | cut -d= -f2)/epsilon-status | jq .; \
	done

# Cleanup targets
clean:
	@echo "$(RED)WARNING: This will remove all containers, volumes, and networks!$(NC)"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	$(COMPOSE_CMD) down -v --remove-orphans
	docker network prune -f
	@echo "$(GREEN)Cleanup complete$(NC)"

clean-volumes:
	@echo "$(YELLOW)Removing persistent volumes...$(NC)"
	$(COMPOSE_CMD) down -v

clean-logs:
	@echo "$(YELLOW)Clearing service logs...$(NC)"
	@for service in $(SERVICES); do \
		docker exec dalrn-$$service sh -c 'rm -rf /app/logs/*' 2>/dev/null || true; \
	done
	@echo "$(GREEN)Logs cleared$(NC)"

# Backup and restore
backup:
	@echo "$(GREEN)Creating backup of DALRN data...$(NC)"
	@mkdir -p backups
	@BACKUP_NAME="dalrn-backup-$$(date +%Y%m%d-%H%M%S)"; \
	docker run --rm \
		-v dalrn_postgres_data:/postgres \
		-v dalrn_redis_data:/redis \
		-v dalrn_ipfs_data:/ipfs \
		-v dalrn_fl_models:/models \
		-v $$(pwd)/backups:/backup \
		alpine tar czf /backup/$$BACKUP_NAME.tar.gz /postgres /redis /ipfs /models
	@echo "$(GREEN)Backup created: backups/$$BACKUP_NAME.tar.gz$(NC)"

restore:
	@echo "$(YELLOW)Available backups:$(NC)"
	@ls -1 backups/*.tar.gz 2>/dev/null || echo "No backups found"
	@echo ""
	@echo "To restore, run: make restore-file FILE=backups/<backup-name>.tar.gz"

restore-file:
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)Error: Please specify backup file with FILE=<path>$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)WARNING: This will overwrite existing data!$(NC)"
	@echo "Restoring from $(FILE)..."
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	$(COMPOSE_CMD) down
	docker run --rm \
		-v dalrn_postgres_data:/postgres \
		-v dalrn_redis_data:/redis \
		-v dalrn_ipfs_data:/ipfs \
		-v dalrn_fl_models:/models \
		-v $$(pwd):/backup \
		alpine tar xzf /backup/$(FILE) -C /
	@echo "$(GREEN)Restore complete$(NC)"

# Development helpers
shell-%:
	@echo "$(GREEN)Opening shell in $* service...$(NC)"
	docker exec -it dalrn-$* /bin/bash

# Quick development cycle
dev: stop-all build run-all logs-follow

# Production deployment
deploy: env-setup validate-config build
	@echo "$(GREEN)Deploying DALRN in production mode...$(NC)"
	$(COMPOSE_CMD) up -d --scale gateway=2 --scale search=2
	@echo "$(GREEN)Production deployment complete$(NC)"
	@make health-check

# Performance testing
perf-test:
	@echo "$(GREEN)Running performance tests with PoDP overhead measurement...$(NC)"
	@echo "Starting load test..."
	@for i in $$(seq 1 100); do \
		curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" http://localhost:8000/health & \
	done; \
	wait
	@echo "$(GREEN)Performance test complete$(NC)"

# Generate PoDP compliance report
podp-report:
	@echo "$(GREEN)Generating PoDP Compliance Report...$(NC)"
	@echo "===================="
	@echo "DALRN PoDP Compliance Report" > podp-report.txt
	@echo "Generated: $$(date)" >> podp-report.txt
	@echo "" >> podp-report.txt
	@for service in $(SERVICES); do \
		echo "Service: $$service" >> podp-report.txt; \
		curl -s http://localhost:$$(grep "$${service^^}_PORT" $(ENV_FILE) | cut -d= -f2)/podp-stats >> podp-report.txt 2>/dev/null || echo "  No PoDP stats available" >> podp-report.txt; \
		echo "" >> podp-report.txt; \
	done
	@echo "$(GREEN)Report saved to podp-report.txt$(NC)"

# Default target
.DEFAULT_GOAL := help