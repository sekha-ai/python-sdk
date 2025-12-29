#!/bin/bash
set -e

echo "📦 Running Sekha Python SDK Test Suite..."

TEST_TYPE=${1:-"all"}

case $TEST_TYPE in
  "lint")
    echo "🔍 Running ruff and black..."
    ruff check .
    black --check .
    ;;
  "unit")
    echo "Running unit tests..."
    pytest tests/ -v
    ;;
  "all"|*)
    echo "Running linting and all tests with coverage..."
    ruff check .
    black --check .
    pytest tests/ -v --cov=sekha --cov-report=html
    ;;
esac

echo "✅ Tests complete!"