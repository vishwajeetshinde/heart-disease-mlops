#!/usr/bin/env python3
"""
Quick Linting Checker
Shows what linting issues exist without fixing them
"""

import subprocess
import sys


def check_tool_installed(tool):
    """Check if a tool is installed"""
    try:
        subprocess.run([tool, '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_linting():
    """Check for linting issues"""
    print("=" * 60)
    print("LINTING CHECKER")
    print("=" * 60)
    
    # Check if flake8 is installed
    if not check_tool_installed('flake8'):
        print("\n❌ flake8 is not installed")
        print("\nInstall with:")
        print("  pip install flake8")
        print("\nOr run the auto-fixer:")
        print("  python fix_linting.py")
        return
    
    print("\n✅ flake8 is installed\n")
    
    # Check each directory
    directories = ['src', 'tests', 'notebooks']
    total_issues = 0
    
    for directory in directories:
        print(f"\n{'=' * 60}")
        print(f"Checking: {directory}/")
        print('=' * 60)
        
        try:
            result = subprocess.run(
                ['flake8', f'{directory}/', '--max-line-length=120',
                 '--ignore=E203,W503', '--statistics', '--count'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ No issues found in {directory}/")
            else:
                print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                # Try to count issues
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.strip() and line[0].isdigit():
                        count = line.split()[0]
                        if count.isdigit():
                            total_issues += int(count)
                            
        except Exception as e:
            print(f"⚠️  Error checking {directory}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if total_issues == 0:
        print("✅ No linting issues found!")
        print("\nYour code is clean and ready to commit!")
    else:
        print(f"⚠️  Found issues in your code")
        print("\n🔧 To fix automatically, run:")
        print("   python fix_linting.py")
        print("\n📖 For manual fixes, see:")
        print("   LINTING_FIXES.md")
    
    print()


if __name__ == "__main__":
    try:
        check_linting()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
