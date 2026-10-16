# SVD residuals and anomaly calibration

For document-term matrix `X=U S V^T`, retain k singular directions. Document
vectors are `U_k S_k`; cosine similarity compares their directions. The
orthogonal projection is `X V_k V_k^T`. Its squared row residual can be computed
as `||x||² - ||x V_k||²`, avoiding a dense reconstructed document-term matrix.
Small negative roundoff is clipped to zero.

Automatic k is the smallest available rank retaining 90% of total squared
singular energy, capped at `min(100,min(shape)-1)` (rank one for singleton axes).
The cap can prevent reaching 90%; it is not a guarantee of retained energy.
Explicit full rank is supported for small matrices to validate reconstruction.

Anomalies use `min(p99(errors), mean(errors)+3*std(errors))`. The default document
batch estimates this on the same corpus, so it is an exploratory ranking rather
than a validated detector. A percentile recomputed on a contaminated batch
typically flags around 1%, not all injected anomalies. Unknown vocabulary and
dominant outlier directions can also hide unusual documents.

The anomaly benchmark trains on a normal synthetic rank-three matrix and
calibrates on a separate normal sample, then evaluates held-out normal and odd
rows. See [the measured CSV](../benchmarks/results/anomaly.csv). Those results
do not establish production document precision/recall. Supply representative
tenant labels and a fixed calibration threshold before using scores as a gate.
