#!/usr/bin/env python3
"""
Automatic Linting Fixer
Fixes common Python linting issues automatically
"""

import subprocess
import sys
import os


def install_tools():
    """Install required linting and formatting tools"""
    print("📦 Installing linting tools...")
    tools = ['flake8', 'autopep8', 'isort']
    
    for tool in tools:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', tool], check=True)
            print(f"  ✅ {tool} installed")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  Failed to install {tool}")
    print()


def fix_imports(directory):
    """Fix import ordering using isort"""
    print(f"🔧 Fixing imports in {directory}...")
    try:
        result = subprocess.run(
            ['isort', directory, '--profile', 'black', '--line-length', '120'],
            capture_output=True,
            text=True
        )
        print(f"  ✅ Imports fixed in {directory}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Error fixing imports: {e}")
        return False
    except FileNotFoundError:
        print("  ⚠️  isort not found, skipping import fixes")
        return False


def fix_pep8(directory):
    """Fix PEP8 issues using autopep8"""
    print(f"🔧 Fixing PEP8 issues in {directory}...")
    try:
        result = subprocess.run(
            ['autopep8', '--in-place', '--recursive', '--aggressive', '--aggressive',
             '--max-line-length', '120', directory],
            capture_output=True,
            text=True
        )
        print(f"  ✅ PEP8 issues fixed in {directory}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Error fixing PEP8: {e}")
        return False
    except FileNotFoundError:
        print("  ⚠️  autopep8 not found, skipping PEP8 fixes")
        return False


def run_flake8_check(directory):
    """Run flake8 to check for remaining issues"""
    print(f"\n🔍 Checking {directory} with flake8...")
    try:
        result = subprocess.run(
            ['flake8', directory, '--max-line-length=120', 
             '--ignore=E203,W503,E501', '--statistics'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  ✅ No linting issues in {directory}")
            return True
        else:
            print(f"  ⚠️  Found issues in {directory}:")
            print(result.stdout)
            return False
    except FileNotFoundError:
        print("  ⚠️  flake8 not found, skipping checks")
        return False


def fix_common_issues_manual():
    """Provide manual fixes for common issues"""
    print("\n📋 Common Manual Fixes:")
    print("-" * 60)
    print("1. Line too long (E501)")
    print("   → Break long lines at 120 characters")
    print("   → Use parentheses for line continuation")
    print()
    print("2. Unused imports (F401)")
    print("   → Remove unused import statements")
    print()
    print("3. Undefined names (F821)")
    print("   → Add missing imports or define variables")
    print()
    print("4. Module imported but unused (F401)")
    print("   → Remove or use the imported module")
    print()
    print("5. Multiple spaces after operator (E221)")
    print("   → Use single space after operators")
    print()
    print("6. Blank line contains whitespace (W293)")
    print("   → Remove trailing whitespace from blank lines")
    print()


def main():
    """Main fixer function"""
    print("=" * 60)
    print("PYTHON LINTING AUTO-FIXER")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('src') and not os.path.exists('tests'):
        print("⚠️  Please run this script from the project root directory")
        return
    
    # Install tools if needed
    install_tools()
    
    # Directories to fix
    directories = []
    if os.path.exists('src'):
        directories.append('src')
    if os.path.exists('tests'):
        directories.append('tests')
    if os.path.exists('notebooks'):
        directories.append('notebooks')
    
    if not directories:
        print("⚠️  No directories found to lint")
        return
    
    # Fix each directory
    for directory in directories:
        print(f"\n{'=' * 60}")
        print(f"Processing: {directory}")
        print('=' * 60)
        
        # Step 1: Fix imports
        fix_imports(directory)
        
        # Step 2: Fix PEP8 issues
        fix_pep8(directory)
        
        # Step 3: Check remaining issues
        run_flake8_check(directory)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Automatic fixes applied")
    print("🔍 Run 'flake8 src/ tests/' to check for remaining issues")
    print()
    
    # Show manual fix guide
    fix_common_issues_manual()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
