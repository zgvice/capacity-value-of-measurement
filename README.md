# Replication

Run from the package root.

## Setup

Use Python 3.10.11. Install Julia 1.10.10 before running Experiment 4.

```sh
python -m pip install -r requirements.txt
julia --startup-file=no --project=code/adequacy/env -e 'using Pkg; Pkg.instantiate()'
```

## Extract, fit and check the Hubei records

```sh
python -B -X utf8 code/extract_hubei_sources.py
python -B -X utf8 code/prepare_data.py
python -B -X utf8 code/core/validate_customer_records.py
```

## Run the experiments

```sh
python -B -X utf8 code/core/check_annual_sources.py
python -B -X utf8 code/exp1.py
python -B -X utf8 code/exp2.py
python -B -X utf8 code/exp3.py
python -B -X utf8 code/exp4.py
python -B -X utf8 code/exp4.py --supplement
```

Add `--check` to an experiment command to check saved results.

## Run the record checks and draw the figures

```sh
python -B -X utf8 code/core/minimum_record_package.py
python -B -X utf8 reproduce_figures.py
```
