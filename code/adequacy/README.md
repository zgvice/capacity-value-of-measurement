# Run the adequacy checks separately

Run from the package root after setup.

```sh
julia --startup-file=no --project=code/adequacy/env code/core/experiment_four_native.jl results/exp4
python -B -X utf8 code/core/experiment_four_outages.py
```
