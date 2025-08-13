# mmesh

`mmesh` is a C library developed by Alexis Naveros, designed to support meshing
algorithms and routines.

The decimation algorithm provides robust mesh simplification, reducing the
number of vertices and triangles while preserving the mesh's overall shape and
features. It is based on the use of quadric error metrics (QEM), a widely
adopted method in geometry processing.  The implementation maintains a queue of
candidate collapses, always selecting the operation with the lowest geometric
error.  It also applies heuristics to maintain important features and prevent
excessive distortion, such as compactness and boundary weighting.

The mesh optimization routines reorder and optimize triangle meshes for
improved vertex cache locality, reducing rendering costs in real-time graphics
pipelines. The core goals are to minimize the Average Cache Miss Rate (ACMR)
and optimize the order of triangle indices for GPU efficiency.
The optimizer approach is inspired by approaches such as Forsyth's and Tipsy's
algorithms but includes custom scoring and parallelization strategies.

