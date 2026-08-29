# Food Security Risk Score

Requires Python 3.9+ and pandas. The FAOSTAT source CSVs live in `faostat/`;
each script reads them from `DATA_DIR`, set to `"faostat"` at the top of the
file, so run the scripts from the project root.

## `load_and_audit.py`

```bash
python3 load_and_audit.py
```

Read-only audit of the five FAOSTAT CSVs.

## `internal_risk_score.py`

```bash
python3 internal_risk_score.py
```

Per-year internal supply metrics from production and trade totals: apparent
domestic supply, self-sufficiency ratio (SSR), import dependency ratio (IDR)
and the internal risk (rolling coefficient of variation of production).
Writes `internal_risk_score.csv`.

## `external_risk_score.py`

```bash
python3 external_risk_score.py
```

Per-year external risk: the Herfindahl-Hirschman index over import suppliers,
with one trade-matrix source (self-reported or mirror) selected per year by
how well its total reconciles with the country import total. Writes
`external_risk_score.csv`.

## Tests

```bash
python3 -m unittest test_internal_risk_score test_external_risk_score test_load_and_audit
```
