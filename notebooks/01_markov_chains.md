# Markov chains and the four-node example

A column-stochastic matrix has nonnegative entries and each column sums to one.
The update `r_next = P @ r` propagates probability from sources (columns) to
destinations (rows). A stationary distribution solves `P @ r = r`: it is the
normalized eigenvector for eigenvalue one. Reducible chains can have several
stationary distributions; periodic chains can oscillate.

Consider A→B, A→C, B→C, C→A, D→C with unit edge weights:

```text
        A    B    C    D
P = A [ 0    0    1    0 ]
    B [ 1/2  0    0    0 ]
    C [ 1/2  1    0    1 ]
    D [ 0    0    0    0 ]
```

Starting with `r0=(1/4,1/4,1/4,1/4)`, the undamped first step is
`P r0=(1/4,1/8,5/8,0)`. At damping 0.85, add 0.15/4 to each coordinate:
`r1=(0.25,0.14375,0.56875,0.0375)`.
The next step is `r2=(0.5209375,0.14375,0.2978125,0.0375)`.

At stationarity, `D=0.0375`, `B=0.425 A+0.0375`,
`A=0.85 C+0.0375`, and `C=0.85(0.5 A+B+D)+0.0375`.
Solving these equations and normalizing agrees with both power iteration and
`numpy.linalg.eig`; the executed values are in [measurements](assets/measurements.json).

The Google matrix is `G=d P + (1-d) 1 1^T/n`. Uniform teleportation makes every
entry positive (after repairing dangling columns), ensuring a unique stationary
distribution and convergence. The magnitude of the second eigenvalue controls
the asymptotic rate: roughly `error_t ~ |lambda_2|^t`. Damping shrinks the
nonprincipal spectrum. Dangling probability must be redistributed before the
next convergence test; adding only a fraction of missing mass breaks normalization.

Reproduce with `python scripts/generate_math_artifacts.py` and
`pytest tests/test_pagerank.py`.
