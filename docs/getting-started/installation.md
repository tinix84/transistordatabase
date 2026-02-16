# Installation

## Requirements

- Python >= 3.10
- pip or conda

## Install from PyPI

```bash
pip install transistordatabase
```

## Install from Source

```bash
git clone https://github.com/tinix84/transistordatabase.git
cd transistordatabase
pip install -e .
```

## Optional Dependencies

### For Web Interface

```bash
pip install transistordatabase[web]
# Or manually:
pip install fastapi uvicorn httpx
```

### For GUI Development

```bash
pip install transistordatabase[gui]
# Or manually:
pip install PyQt5
```

### For Documentation

```bash
pip install transistordatabase[docs]
# Or manually:
pip install sphinx sphinx-rtd-theme mkdocs mkdocs-material mkdocstrings[python]
```

## Verify Installation

```python
import transistordatabase
print(transistordatabase.__version__)
```

## Troubleshooting

If you encounter issues during installation, please check:

1. Python version >= 3.10
2. All dependencies installed correctly
3. [GitHub Issues](https://github.com/tinix84/transistordatabase/issues)
