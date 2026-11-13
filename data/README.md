# Prepare the data

Run from the package root.

```sh
python -B -X utf8 code/extract_hubei_sources.py
python -B -X utf8 code/prepare_data.py
python -B -X utf8 code/core/validate_customer_records.py
```
