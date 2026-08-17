# Manuscript tables

```powershell
python paper_tables/generate_tables.py 2
python paper_tables/generate_tables.py 3
python paper_tables/generate_tables.py all
```

- `table_02_convergence.csv`: discretization and active-constraint verification.
- `table_03_policy_comparison.csv`: constant-D and DNN policy comparison.

Table 3 uses a direct temporal B-spline PSO with independently initialized
particles. It does not seed from or fall back to the maximal-feasible policy.
