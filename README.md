# Food Security Risk Score

Requires Python 3.9+ and pandas. The FAOSTAT source CSVs live in `faostat/`;
each script reads them from `DATA_DIR`, set to `"faostat"` at the top of the
file, so run the scripts from the project root.

## `load_and_audit.py`

```bash
python3 load_and_audit.py
```

Read-only audit of the five FAOSTAT CSVs.

## `food_risk_score.py`

```bash
python3 food_risk_score.py
```
