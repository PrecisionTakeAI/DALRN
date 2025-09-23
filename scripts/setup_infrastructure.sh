#!/bin/bash
# DALRN Infrastructure Setup Script

set -e

echo "========================================="
echo "DALRN Infrastructure Setup"
echo "========================================="

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo "❌ Docker Compose is not installed. Please install Docker Compose."
        exit 1
    fi

    echo "✅ Docker and Docker Compose are installed"
}

# Function to check if services are healthy
check_health() {
    local service=$1
    local max_attempts=30
    local attempt=1

    echo -n "Waiting for $service to be healthy..."

    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps $service | grep -q "healthy"; then
            echo " ✅"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done

    echo " ⚠️ (timeout)"
    return 1
}

# Function to start services
start_services() {
    echo ""
    echo "Starting infrastructure services..."

    # Start core services first
    docker-compose up -d postgres redis

    # Wait for database to be ready
    check_health postgres
    check_health redis

    # Start IPFS
    docker-compose up -d ipfs
    check_health ipfs

    # Start blockchain
    docker-compose up -d anvil
    check_health anvil

    # Start monitoring (optional)
    if [ "$1" == "--with-monitoring" ]; then
        echo "Starting monitoring services..."
        docker-compose up -d prometheus grafana
    fi

    # Start management tools (optional)
    if [ "$1" == "--with-tools" ]; then
        echo "Starting management tools..."
        docker-compose --profile tools up -d
    fi
}

# Function to show service status
show_status() {
    echo ""
    echo "Service Status:"
    echo "----------------------------------------"
    docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
}

# Function to show connection info
show_connection_info() {
    echo ""
    echo "Connection Information:"
    echo "----------------------------------------"
    echo "PostgreSQL:"
    echo "  Host: localhost"
    echo "  Port: 5432"
    echo "  Database: dalrn"
    echo "  User: dalrn_user"
    echo "  Password: changeme"
    echo ""
    echo "Redis:"
    echo "  Host: localhost"
    echo "  Port: 6379"
    echo ""
    echo "IPFS:"
    echo "  API: http://localhost:5001"
    echo "  Gateway: http://localhost:8080"
    echo "  WebUI: http://localhost:5001/webui"
    echo ""
    echo "Ethereum (Anvil):"
    echo "  RPC URL: http://localhost:8545"
    echo "  Chain ID: 31337"
    echo ""

    if [ "$1" == "--with-monitoring" ]; then
        echo "Prometheus:"
        echo "  URL: http://localhost:9090"
        echo ""
        echo "Grafana:"
        echo "  URL: http://localhost:3000"
        echo "  User: admin"
        echo "  Password: admin"
        echo ""
    fi

    if [ "$1" == "--with-tools" ]; then
        echo "pgAdmin:"
        echo "  URL: http://localhost:5050"
        echo "  Email: admin@dalrn.local"
        echo "  Password: admin"
        echo ""
        echo "Redis Commander:"
        echo "  URL: http://localhost:8081"
        echo ""
    fi
}

# Function to test connections
test_connections() {
    echo ""
    echo "Testing Connections:"
    echo "----------------------------------------"

    # Test PostgreSQL
    if docker exec dalrn-postgres pg_isready -U dalrn_user -d dalrn &> /dev/null; then
        echo "✅ PostgreSQL is accessible"
    else
        echo "❌ PostgreSQL connection failed"
    fi

    # Test Redis
    if docker exec dalrn-redis redis-cli ping &> /dev/null; then
        echo "✅ Redis is accessible"
    else
        echo "❌ Redis connection failed"
    fi

    # Test IPFS
    if curl -s http://localhost:5001/api/v0/version &> /dev/null; then
        echo "✅ IPFS API is accessible"
    else
        echo "❌ IPFS API connection failed"
    fi

    # Test Anvil
    if curl -s -X POST http://localhost:8545 \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' &> /dev/null; then
        echo "✅ Ethereum RPC is accessible"
    else
        echo "❌ Ethereum RPC connection failed"
    fi
}

# Main script
main() {
    check_docker

    # Parse arguments
    MONITORING=""
    TOOLS=""
    for arg in "$@"; do
        case $arg in
            --with-monitoring)
                MONITORING="--with-monitoring"
                ;;
            --with-tools)
                TOOLS="--with-tools"
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --with-monitoring  Start Prometheus and Grafana"
                echo "  --with-tools       Start pgAdmin and Redis Commander"
                echo "  --help            Show this help message"
                exit 0
                ;;
        esac
    done

    # Create necessary directories
    mkdir -p infra/prometheus
    mkdir -p infra/grafana/provisioning/datasources
    mkdir -p infra/grafana/provisioning/dashboards
    mkdir -p infra/grafana/dashboards
    mkdir -p scripts

    # Start services
    start_services $MONITORING $TOOLS

    # Show status
    show_status

    # Test connections
    test_connections

    # Show connection info
    show_connection_info $MONITORING $TOOLS

    echo ""
    echo "========================================="
    echo "✅ Infrastructure setup complete!"
    echo "========================================="
}

# Run main function
main "$@"