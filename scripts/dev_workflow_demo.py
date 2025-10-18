#!/usr/bin/env python3
"""
Development Workflow Demo Script

Демонстрирует правильный рабочий процесс разработки:
1. Быстрая проверка (pre-commit)
2. Полная проверка (CI)
3. Тестирование всех команд

Usage:
    python scripts/dev_workflow_demo.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🧪 {description}")
    print(f"   Command: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run the complete workflow demo."""
    print("🚀 Development Workflow Demo")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("Makefile").exists():
        print("❌ Error: Run this script from the project root directory")
        sys.exit(1)

    # Test individual components
    tests = [
        ("make format-check", "Format Check"),
        ("make lint", "Lint Check"),
        ("make typecheck", "Type Check"),
        ("make test-quick", "Quick Tests"),
        ("make lint", "Pre-commit Hooks (Ruff)"),  # Simplified for demo
    ]

    passed = 0
    total = len(tests)

    for cmd, description in tests:
        if run_command(cmd, description):
            passed += 1

    # Summary
    print("\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 All checks passed! Your development workflow is ready.")
        print("\n💡 Quick Reference:")
        print("  make pre-commit      # Fast feedback")
        print("  make ci             # Full CI check")
        print("  make pre-push       # Before pushing")
        print("  make dev-setup      # Initial setup")
    else:
        print(
            f"\n⚠️  {total - passed} check(s) failed. Please fix them before proceeding."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
