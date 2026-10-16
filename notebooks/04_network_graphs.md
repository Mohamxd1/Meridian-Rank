# Routing, centrality, and failure impact

Dijkstra minimizes the sum of positive OSPF-style costs. The distance-vector
simulation starts with self and direct-neighbor distances. Each synchronous
exchange adds paths one edge longer. The returned count includes the final
unchanged exchange: a ten-node ring takes five rounds; a leaf-spine fabric with
hosts has diameter four and takes four. Executed examples are in
[measurements](assets/measurements.json).

Weighted Brandes centrality accumulates fractions of minimum-cost paths through
each node. For undirected graphs, unnormalized counts are halved; normalized
counts divide ordered-pair totals by `(n-1)(n-2)`. Eigenvector centrality uses
unit adjacency and an `A+I` iteration to prevent bipartite oscillation. PageRank
uses the same unit edges with teleportation and always sums to one.

Failure impact counts pairs that were reachable and became unreachable, including
pairs involving a removed endpoint. Directed graphs count ordered pairs;
undirected graphs count unordered pairs. Components are weak components. Stretch
is averaged only over pairs reachable before and after; when none survive it is
defined as one. Loads aggregate both directions of a physical undirected link.

The illustrative infrastructure YAML is a design example, not a verified live
inventory. The generated ranking describes structural dependency in that model,
not production availability or observed incidents.
