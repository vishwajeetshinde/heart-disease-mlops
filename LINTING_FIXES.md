# Linting Issues - Quick Fix Guide

## Quick Fix Options

### Option 1: Automatic Fix Script (Recommended)
```bash
# Run the auto-fixer
python fix_linting.py
```

This will:
- Install required tools (flake8, autopep8, isort)
- Fix import ordering
- Fix PEP8 issues automatically
- Show remaining issues

### Option 2: Manual Fixes

#### Install Tools First
```bash
pip install flake8 autopep8 isort
```

#### Fix Imports
```bash
isort src/ tests/ notebooks/ --profile black --line-length 120
```

#### Fix PEP8 Issues
```bash
autopep8 --in-place --recursive --aggressive --aggressive src/ tests/ notebooks/
```

#### Check Remaining Issues
```bash
flake8 src/ tests/ notebooks/ --max-line-length=120 --ignore=E203,W503
```

### Option 3: Disable Strict Linting in CI/CD

If you want to make linting non-blocking, update `.github/workflows/ci.yml`:

```yaml
- name: Lint with flake8
  run: |
    flake8 src/ tests/ --max-line-length=120 --ignore=E203,W503 || true
```

The `|| true` makes it always pass even with warnings.

## Common Linting Errors & Fixes

### E501 - Line too long
**Error:** Line exceeds 120 characters

**Fix:**
```python
# Before
result = some_function(parameter1, parameter2, parameter3, parameter4, parameter5, parameter6)

# After
result = some_function(
    parameter1, parameter2, parameter3,
    parameter4, parameter5, parameter6
)
```

### F401 - Module imported but unused
**Error:** Import statement not used in code

**Fix:** Remove the unused import
```python
# Before
import pandas as pd
import numpy as np  # Not used

# After
import pandas as pd
```

### E302 - Expected 2 blank lines
**Error:** Missing blank lines between functions/classes

**Fix:**
```python
# Before
def function1():
    pass
def function2():
    pass

# After
def function1():
    pass


def function2():
    pass
```

### W293 - Blank line contains whitespace
**Error:** Blank lines have spaces/tabs

**Fix:** Remove all whitespace from blank lines

### E231 - Missing whitespace after ','
**Error:** Missing space after comma

**Fix:**
```python
# Before
my_list = [1,2,3,4]

# After
my_list = [1, 2, 3, 4]
```

### E225 - Missing whitespace around operator
**Error:** Missing spaces around operators

**Fix:**
```python
# Before
x=5+3

# After
x = 5 + 3
```

## Configuration Files Already Set Up

### .flake8
- Max line length: 120
- Ignores: E203, W503, E501
- Excludes: .git, __pycache__, mlruns, etc.

### pytest.ini
- Test discovery patterns configured
- Coverage settings configured

## Quick Commands Reference

```bash
# Check linting issues
flake8 src/ tests/

# Fix automatically
autopep8 --in-place --recursive src/ tests/

# Fix imports
isort src/ tests/

# Run tests (will also check linting in CI)
pytest tests/ -v

# Full CI/CD simulation locally
python run_tests.py
```

## Ignoring Specific Lines

If you need to ignore a specific line:

```python
# Ignore specific error
result = very_long_line_that_cannot_be_shortened()  # noqa: E501

# Ignore all errors on line
some_code()  # noqa

# Ignore entire file (add at top)
# flake8: noqa
```

## VS Code Integration (Optional)

If using VS Code, add to `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": [
    "--max-line-length=120",
    "--ignore=E203,W503"
  ],
  "python.formatting.provider": "autopep8",
  "python.formatting.autopep8Args": [
    "--max-line-length=120"
  ],
  "[python]": {
    "editor.formatOnSave": true
  }
}
```

## Troubleshooting

### "flake8: command not found"
```bash
pip install flake8
# or
python -m pip install flake8
```

### "autopep8: command not found"
```bash
pip install autopep8
```

### Too many errors to fix manually
```bash
# Use the auto-fixer script
python fix_linting.py

# Or update CI to be less strict (add || true)
```

### Linting passes locally but fails in CI
- Ensure same Python version
- Check .flake8 configuration is committed
- Verify requirements.txt has flake8

## Best Practices

1. ✅ Run linting before committing
2. ✅ Use automatic formatters (autopep8, black)
3. ✅ Configure editor to format on save
4. ✅ Keep line length under 120 characters
5. ✅ Remove unused imports regularly
6. ✅ Follow PEP 8 style guide

---

**Need Help?** Run `python fix_linting.py` for automatic fixes!
