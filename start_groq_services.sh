#!/bin/bash
echo "Starting Groq LPU Services..."
echo ""

# Load environment variables
if [ -f .env.groq ]; then
    export $(cat .env.groq | xargs)
fi

# Start services
echo "Starting Groq Search Service on port 9001..."
python -m services.search.groq_search_service &

echo "Starting Groq FHE Service on port 9002..."
python -m services.fhe.groq_fhe_service &

echo ""
echo "Groq services started!"
echo "Check http://localhost:9001/health for search service"
echo "Check http://localhost:9002/health for FHE service"
