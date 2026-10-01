
# Databricks Setup

## 1. Configure the Databricks environment

Open the project's root `pyproject.toml` in the Databricks Git Folder editor.

Add:

[tool.databricks.environment]
environment_version = "6"

Save the file and click **Apply** to apply the updated environment.

---

## 2. Open a notebook

Open or create a notebook in the Databricks workspace.

Change the working directory to the root of the Git Folder:

```python
%cd /Workspace/Users/<your-workspace-user>/finance-timeseries-ml-platform
```

Verify that the project file is accessible:

```python
import os

print(os.getcwd())
print(os.path.exists("pyproject.toml"))
```

The second command should return:

```text
True
```

---

## 3. Synchronize project dependencies

Run:

```python
%uv sync
```

This installs the dependencies declared in `pyproject.toml` and creates or updates `uv.lock`.

---

## 4. Restart the Python kernel

After synchronizing the environment, restart the Python kernel:

```python
%restart_python
```

Run `%restart_python` in a cell by itself.

After the restart, the notebook can use the updated project environment and dependencies.

---

## 5. Verify the environment

Optionally verify the main package versions:

```python
import sys
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import pyspark

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("PySpark:", pyspark.__version__)
```
```