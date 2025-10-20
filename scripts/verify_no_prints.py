"""Verify no print() statements in production code.

This script checks that all print() statements have been replaced with structured logging.
"""

import re
import sys
from pathlib import Path


def check_file_for_prints(file_path: Path) -> list[tuple[int, str]]:
    """Check file for print() statements.

    Args:
        file_path: Path to file to check

    Returns:
        List of (line_number, line_content) tuples for lines with print()
    """
    violations = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if '"""' in stripped or "'''" in stripped:
                continue

            # Check for print() statements
            if re.search(r"\bprint\s*\(", line):
                violations.append((line_num, line.rstrip()))

    return violations


def main() -> int:
    """Check all production code for print() statements.

    Returns:
        Exit code (0 = success, 1 = violations found)
    """
    # Directories to check
    src_dir = Path("src")
    scripts_dir = Path("scripts")

    # Files to exclude (test files, this script, etc.)
    exclude_patterns = [
        "test_*.py",
        "*_test.py",
        "conftest.py",
        "verify_no_prints.py",
        "diagnose_*.py",
        "demo_*.py",
    ]

    violations_found = False

    # Check src/ directory
    print(f"Checking {src_dir}/ for print() statements...")
    for py_file in src_dir.rglob("*.py"):
        # Skip excluded files
        if any(py_file.match(pattern) for pattern in exclude_patterns):
            continue

        violations = check_file_for_prints(py_file)
        if violations:
            violations_found = True
            print(f"\n❌ {py_file}:")
            for line_num, line_content in violations:
                print(f"   Line {line_num}: {line_content}")

    # Check scripts/ directory (only pipeline scripts)
    pipeline_scripts = [
        scripts_dir / "run_multi_source_pipeline.py",
        scripts_dir / "run_pipeline.py",
        scripts_dir / "generate_digest.py",
        scripts_dir / "backfill.py",
    ]

    print("\nChecking pipeline scripts for print() statements...")
    for py_file in pipeline_scripts:
        if not py_file.exists():
            continue

        violations = check_file_for_prints(py_file)
        if violations:
            violations_found = True
            print(f"\n❌ {py_file}:")
            for line_num, line_content in violations:
                print(f"   Line {line_num}: {line_content}")

    if violations_found:
        print("\n❌ Print statements found in production code!")
        print("   Please replace with structured logging (logger.info/warning/error)")
        return 1
    else:
        print("\n✅ No print() statements found in production code!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
