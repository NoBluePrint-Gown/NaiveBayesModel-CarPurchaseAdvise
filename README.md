# Car Evaluation AI System

A single-file machine learning system that predicts car conditions using a Naive Bayes classifier.

## What Is This System?

The **Car Evaluation AI System** predicts the overall condition of a used car based on six key attributes:

| Attribute | Description | Possible Values |
|-----------|-------------|-----------------|
| **buying** | Buying price | vhigh, high, med, low |
| **maint** | Maintenance cost | vhigh, high, med, low |
| **doors** | Number of doors | 2, 3, 4, 5more |
| **persons** | Passenger capacity | 2, 4, more |
| **lug_boot** | Luggage boot size | small, med, big |
| **safety** | Safety rating | low, med, high |

**Output conditions:**
- **Unacceptable** (unacc) - Don't buy
- **Acceptable** (acc) - Reasonable choice
- **Good** (good) - Solid option
- **Very Good** (vgood) - Excellent choice

## Features

- **Single file** - Everything in `main.py`
- **Zero dependencies** - Only requires `scikit-learn`
- **No file I/O** - Model stays in memory, no pickle/joblib files
- **Trains on startup** - Always fresh model
- **Shows confidence** - Displays prediction probabilities

## Requirements

- Python 3.6 or higher
- scikit-learn

## Installation (Linux/Mac)

```bash
# Install scikit-learn
pip install scikit-learn

# Or with --user if permission issues
pip install --user scikit-learn
```
## Installation (Windows)

```bash
# Install scikit-learn
py -m pip install scikit-learn

# Or with --user if permission issues
pip -m pip install --user scikit-learn
```

## Execution

```bash
# Use what ever environment variable is set (py, python, python3, etc...)
py main.py
```
