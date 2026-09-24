# Data Architecture

This project follows a simplified lakehouse architecture using Bronze, Silver, and Gold data layers.

## Overview

```text
Synthetic financial process
        ↓
Controlled data-quality corruption
        ↓
Bronze
        ↓
Cleaning and validation
        ↓
Silver
        ↓
Feature engineering
        ↓
Gold
        ↓
Model training / inference
        ↓
Predictions
        ↓
Power BI