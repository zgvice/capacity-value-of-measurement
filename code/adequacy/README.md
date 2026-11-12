# External adequacy checks

Run the commands below from the replication package root. The two checks use different inputs and answer different questions.

**Matched annual shortfalls.** The Julia program passes the bundled rounded load, supply and response paths to the Probabilistic Resource Adequacy Suite (PRAS). It checks every five-minute interval in each of the twelve annual cases. Install Julia 1.10.10 and instantiate the supplied project once. Julia may need internet access to install the registered dependencies.

```sh
julia --startup-file=no --project="code/adequacy/env" -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project="code/adequacy/env" "code/core/experiment_four_native.jl" "results/exp4"
```

The second command reads `pras_input.csv` and writes `pras_results.json` in the supplied results directory. `pras_comparison.csv` (D24) gives the paper's comparison and also reports rounding differences against the unrounded study inputs.

**Generator outage dependence.** The Python program reads the RTS-GMLC test system supplied with PRAS and varies the dependence of generator outages while retaining the marginal failure and repair probabilities. The simulation settings are in `experiment_config.json` (I18).

```sh
python -B -X utf8 "code/core/experiment_four_outages.py"
```

This writes `outage_dependence.csv` (D26) and `outage_params.json` under `results/exp4`.

The supplied PRAS source is version 0.8.0 at [revision 11d3003e0ff3d343f278cf4f6a1c0d85193fb](https://github.com/NatLabRockies/PRAS/tree/11d3003e0ff3d343f278cf4f6a1c0d85193fb). Its module directories contain the original MIT licenses. The Julia manifest points to these relative paths.
