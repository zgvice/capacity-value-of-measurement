# Submission data

Customer_Labels.csv gives the customer industries and their corresponding single-customer LSE cases. The codes in the observed tables are submission identifiers. Portfolio codes A+B, A+C, B+C and A+B+C identify customer mixes submitted by LSE D. A, B and C stand for the specialty steel, cement and building materials profiles. In portfolio data, member keys such as LSE A refer to the corresponding customer profile. The Normal reference is a simulation benchmark.

D01 contains 743 15-minute observations across 28 customer events on 19 Hubei call days in 2024. The event dates, times, observed loads, targets and other numerical fields are retained at the original precision. Source S codes identify source documents within this extract. The actual-to-target consumption ratios and within-event time order underlie D02 and I12. The records originate from instructions to increase consumption.

D33 and D40 report the whole-event holdout. D34 and D35 report the paired meter and baseline observations and their segment diagnostics. The other numbered files are model inputs or simulation results. The event plot files and Sensitivity_Parameters.json are the data for the accompanying figures.

See ../replication_index.csv for dataset descriptions and paths relative to the replication package root. Original settlement and event workbooks and the internal all-source reconciliation are held in the study archive.
