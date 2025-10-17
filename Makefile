.PHONY: help install format lint typecheck test ci clean

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code with ruff...$(NC)"
	ruff format .
	@echo "$(GREEN)✓ Formatting complete$(NC)"

format-check: ## Check code formatting without modifying files
	@echo "$(BLUE)Checking code formatting...$(NC)"
	@ruff format --check . && echo "$(GREEN)✓ Format check passed$(NC)" || (echo "$(RED)✗ Format check failed$(NC)" && exit 1)

lint: ## Run ruff linter
	@echo "$(BLUE)Running ruff linter...$(NC)"
	@ruff check . && echo "$(GREEN)✓ Lint passed$(NC)" || (echo "$(RED)✗ Lint failed$(NC)" && exit 1)

lint-fix: ## Run ruff linter with auto-fix
	@echo "$(BLUE)Running ruff linter with auto-fix...$(NC)"
	ruff check . --fix --unsafe-fixes
	@echo "$(GREEN)✓ Lint fixes applied$(NC)"

typecheck: ## Run mypy type checker
	@echo "$(BLUE)Running mypy type checker...$(NC)"
	@mypy --strict src tests && echo "$(GREEN)✓ Type check passed$(NC)" || (echo "$(RED)✗ Type check failed$(NC)" && exit 1)

test: ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	@SLACK_BOT_TOKEN=dummy OPENAI_API_KEY=dummy pytest -v && echo "$(GREEN)✓ Tests passed$(NC)" || (echo "$(RED)✗ Tests failed$(NC)" && exit 1)

test-quick: ## Run tests without coverage
	@echo "$(BLUE)Running quick tests...$(NC)"
	@SLACK_BOT_TOKEN=dummy OPENAI_API_KEY=dummy pytest -q --no-cov && echo "$(GREEN)✓ Tests passed$(NC)" || (echo "$(RED)✗ Tests failed$(NC)" && exit 1)

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@SLACK_BOT_TOKEN=dummy OPENAI_API_KEY=dummy pytest --cov=src --cov-report=term-missing --cov-report=html && echo "$(GREEN)✓ Tests passed with coverage$(NC)" || (echo "$(RED)✗ Tests failed$(NC)" && exit 1)

test-postgres: ## Run PostgreSQL tests (requires PostgreSQL running and TEST_POSTGRES=1)
	@echo "$(BLUE)Running PostgreSQL tests...$(NC)"
	@if [ -z "$$POSTGRES_PASSWORD" ]; then \
		echo "$(RED)✗ POSTGRES_PASSWORD not set$(NC)"; \
		echo "$(YELLOW)Set POSTGRES_PASSWORD and TEST_POSTGRES=1 to run PostgreSQL tests$(NC)"; \
		exit 1; \
	fi
	@if [ "$$TEST_POSTGRES" != "1" ]; then \
		echo "$(RED)✗ TEST_POSTGRES not set to 1$(NC)"; \
		echo "$(YELLOW)Set TEST_POSTGRES=1 to run PostgreSQL tests$(NC)"; \
		exit 1; \
	fi
	@SLACK_BOT_TOKEN=dummy OPENAI_API_KEY=dummy pytest tests/test_postgres_repository.py -v && echo "$(GREEN)✓ PostgreSQL tests passed$(NC)" || (echo "$(RED)✗ PostgreSQL tests failed$(NC)" && exit 1)

ci: format-check lint typecheck test ## Run all CI checks (format, lint, typecheck, test)
	@echo "$(GREEN)✓ All CI checks passed!$(NC)"

ci-local: ## Run CI checks locally (same as GitHub Actions)
	@echo "$(BLUE)===========================================$(NC)"
	@echo "$(BLUE)   Running CI Pipeline (Local)$(NC)"
	@echo "$(BLUE)===========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Step 1/4: Format Check$(NC)"
	@make format-check
	@echo ""
	@echo "$(YELLOW)Step 2/4: Lint$(NC)"
	@make lint
	@echo ""
	@echo "$(YELLOW)Step 3/4: Type Check$(NC)"
	@make typecheck
	@echo ""
	@echo "$(YELLOW)Step 4/4: Tests$(NC)"
	@make test
	@echo ""
	@echo "$(GREEN)===========================================$(NC)"
	@echo "$(GREEN)   ✓ CI Pipeline Passed!$(NC)"
	@echo "$(GREEN)===========================================$(NC)"

clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf **/__pycache__
	rm -rf **/*.pyc
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

pre-commit: format-check lint typecheck ## Run pre-commit checks (fast)
	@echo "$(GREEN)✓ Pre-commit checks passed!$(NC)"

pre-push: ci-local ## Run pre-push checks (full CI)
	@echo "$(GREEN)✓ Pre-push checks passed! Safe to push.$(NC)"
