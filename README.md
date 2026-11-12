# Replication materials for The Capacity Value of Measurement

This package contains the inputs, calculation code, and saved numerical results for Section 6 and Appendix B. The dataset and procedure identifiers in the paper refer to [replication_index.csv](replication_index.csv). The paths in that index are relative to this directory.

The Hubei inputs contain the three industrial customers used to fit the simulation profiles. Their records retain the original dates, interval values and numerical precision. The customer and simulation labels are listed in [Customer_Labels.csv](data/Customer_Labels.csv). These records support the empirical power profiles and the baseline and meter diagnostics. The customer histories used in the PJM scenarios are simulated. The public PJM planning workbooks and the frozen annual inputs are in `inputs/`. Source workbooks containing customer identities and the internal reconciliation of all original customers are held separately.

## Setup and figures

From this directory, use Python 3.10.11 and install the versions in `requirements.txt`.

```sh
python -m pip install -r requirements.txt
python -B -X utf8 reproduce_figures.py
```

The second command draws the six numerical figures in `figures/` using the supplied result tables. Figures 2 and 3 read the continuation and record-verification results under `results/`. Figures 4, 5, B.1 and B.2 use the corresponding submission tables in `data/`. Figure 1 is drawn in the manuscript source.

## Calculations

Use a working copy of the package for calculation. The programs write outputs under `results/` and checks under `checks/`. Run the stages in order when recalculating the shared settings and all four experiments.

```sh
python -B -X utf8 "code/prepare_data.py"
python -B -X utf8 "code/core/check_annual_sources.py"
python -B -X utf8 "code/exp1.py"
python -B -X utf8 "code/exp2.py"
python -B -X utf8 "code/exp3.py"
python -B -X utf8 "code/exp4.py"
python -B -X utf8 "code/exp4.py" --supplement
```

- **Data preparation** fits the three customer profiles from the anonymized interval data and checks the frozen inputs. The next command verifies the annual paths against the three public planning workbooks. The retained PJM call instructions and customer calendar are supplied as fixed inputs.
- **Experiment 1** holds historical records fixed and changes continuation capability. **Experiment 2** uses the same customer to compare how records are verified and which records are supplied.
- **Experiment 3** evaluates the individual and aggregate cases, including the evidence sequences. **Experiment 4** runs the five-date holdout, independent annual risk, and physical checks. The supplement entry completes missing sensitivity and diagnostic outputs and reuses matching saved results. The individual scripts are listed under the corresponding results in the index.
- For Experiment 4, install Julia 1.10.10 and instantiate the bundled PRAS project first. See the [external-check instructions](code/adequacy/README.md). The program uses Julia on the system path or the executable specified by the `JULIA` environment variable.

Add `--check` to an experiment entry point for its shorter validation mode. Experiment 1 recomputes risk at the reported capacity bounds. Experiment 2 reruns historical feasibility and checks the saved risk brackets. Experiments 3 and 4 also check saved results. Use the full commands above for new annual calculations. Full runs can be substantially longer than the checks. Set `NUMBA_NUM_THREADS` before starting to limit the thread count.

To reproduce the whole-event holdout and paired baseline diagnostics or the four-record certificate separately, use:

```sh
python -B -X utf8 "code/core/validate_customer_records.py"
python -B -X utf8 "code/core/minimum_record_package.py"
```

The second command tests which subsets of the existing call library can establish the target duration. It reads the capacity mapping saved by Experiment 2. It writes the historical certificate and its capacity comparison under `checks/`.

## Release checks

The six numerical figures were regenerated and compared with the manuscript figures. The released records reproduce the empirical support points, frequencies, and adjacent-period rank correlations. The event holdout and segment diagnostics were rerun. The official annual inputs were reconstructed from the public workbooks, the Experiment 1 risk brackets were checked, and the minimum record certificate was recomputed. The stage checks for Experiments 2--4 also passed. The historical coverage simulation for the five-date holdout matched the reported result. The PRAS matched-path comparison and the RTS-GMLC outage dependence experiment were rerun against the supplied inputs.

The release checks cover these calculations and the supplied results. They do not constitute a new full run of all annual capacity searches and parameter sweeps.

