#!/usr/bin/env python3
"""
Test Runner Script
Runs all tests with coverage reporting
"""

import subprocess
import sys


def run_linting():
    """Run flake8 linting"""
    print("=" * 60)
    print("RUNNING LINTING (flake8)")
    print("=" * 60)
    
    result = subprocess.run(
        ["flake8", "src/", "tests/", "notebooks/"],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("\n⚠️  Linting found issues")
        return False
    else:
        print("\n✅ Linting passed")
        return True


def run_tests():
    """Run all tests with coverage"""
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS (pytest)")
    print("=" * 60)
    
    result = subprocess.run(
        ["pytest", "tests/", "-v", "--cov=src", "--cov=notebooks", 
         "--cov-report=term-missing", "--cov-report=html"],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("\n❌ Tests failed")
        return False
    else:
        print("\n✅ All tests passed")
        return True


def run_specific_test_file(filename):
    """Run a specific test file"""
    print(f"\n{'=' * 60}")
    print(f"RUNNING TEST FILE: {filename}")
    print("=" * 60)
    
    result = subprocess.run(
        ["pytest", f"tests/{filename}", "-v"],
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    """Main test runner"""
    print("\n" + "=" * 60)
    print("HEART DISEASE MLOPS - TEST SUITE")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--lint-only":
            run_linting()
            return
        elif sys.argv[1] == "--test-only":
            run_tests()
            return
        elif sys.argv[1].startswith("test_"):
            run_specific_test_file(sys.argv[1])
            return
        elif sys.argv[1] == "--help":
            print("\nUsage:")
            print("  python run_tests.py              # Run linting + all tests")
            print("  python run_tests.py --lint-only  # Run only linting")
            print("  python run_tests.py --test-only  # Run only tests")
            print("  python run_tests.py test_*.py    # Run specific test file")
            return
    
    # Run both linting and tests
    lint_passed = run_linting()
    test_passed = run_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Linting: {'✅ PASSED' if lint_passed else '❌ FAILED'}")
    print(f"Tests:   {'✅ PASSED' if test_passed else '❌ FAILED'}")
    
    if lint_passed and test_passed:
        print("\n🎉 All checks passed!")
        print("\nCoverage report saved to: htmlcov/index.html")
        sys.exit(0)
    else:
        print("\n⚠️  Some checks failed. Please review and fix.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        sys.exit(1)
