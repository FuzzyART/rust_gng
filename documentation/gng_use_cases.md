88f 
🌍 Where GNG is used because it is uniquely strong
1. Topological reconstruction of complex manifolds
GNG can recover the shape of a dataset — not just clusters.

Why GNG wins:
Learns topology without predefining structure

Handles branching, loops, holes

Works in high dimensions

Incremental and online

Real applications:
Mapping the shape of protein folding landscapes

Reconstructing robot configuration spaces

Modeling nonlinear manifolds in physics simulations

No other classical algorithm does this as reliably.

2. Robotics motion planning
GNG is used to build roadmaps of feasible robot movements.

Why GNG wins:
Learns reachable space incrementally

Adapts as the robot explores

Produces a graph structure directly usable for planning

Real applications:
Autonomous robot navigation

High‑DOF arm motion planning

Exploration in unknown environments

PRM and RRT struggle with incremental refinement; GNG does not.

3. Adaptive vector quantization for compression
GNG is used in:

Image compression

Audio compression

Sensor data compression

Why GNG wins:
Learns codebooks without fixed size

Adapts to non‑stationary data

Preserves topology → fewer artifacts

Better than k‑means, especially for streaming or non‑stationary signals.

4. Surface reconstruction from point clouds
Used in 3D scanning, SLAM, and CAD.

Why GNG wins:
Builds a mesh‑like graph directly

Handles uneven sampling

Robust to noise

No need to predefine mesh resolution

This is why GNG appears in:

3D scanning pipelines

Archaeological reconstruction

Industrial inspection

5. Anomaly detection in streaming data
GNG adapts to the “normal manifold” and flags deviations.

Why GNG wins:
Online learning

No fixed cluster count

Topology changes reveal anomalies

Used in:

Network intrusion detection

Industrial sensor monitoring

Financial fraud detection

6. Exploratory data analysis for unknown structures
When you don’t know:

how many clusters exist

what shape they have

whether the data is continuous or discrete

GNG is one of the few algorithms that can reveal:

cluster count

cluster shape

cluster connectivity

Better than SOM because SOM has a fixed grid.
Better than DBSCAN because DBSCAN cannot handle varying density.
Better than k‑means because k‑means assumes convex clusters.

7. Neuroscience modeling
GNG is used to model:

cortical map formation

sensory topographic maps

neural development processes

Why GNG wins:
It is biologically plausible

It grows like real neural structures

It adapts continuously

SOM is too rigid; GNG is dynamic.

🏆 Summary: When GNG is the best tool
GNG is uniquely strong when you need:

Need	Why GNG wins
Topology learning	Learns manifold shape, not just clusters
Incremental / streaming learning	No retraining needed
Unknown number of clusters	Grows automatically
Graph output	Produces a usable network structure
Nonlinear, branching structures	Handles complexity SOM/k‑means cannot
Noise‑robust surface reconstruction	Excellent for 3D data



# quick notes

sh scripts/2-build_py_lib.sh 
sh scripts/container/1-buildContainer.sh 
sh scripts/container/2-startContainer.sh 
sh scripts/container/3-install_gng.sh 