# NVIDIA cuOpt
Carlos Izquierdo Hernández's research fellowship project at the Department of Industrial Management, Business Administration and Statistics of Universidad Politécnica de Madrid (UPM).

# Overview
This project develops a comparative study applied to Mixed Integer Linear Programming (MILP) and Vehicle Routing Problems (VRP):
- Solver comparison: Traditional CPU-based solvers (CBC and Gurobi) vs. NVIDIA cuOpt evaluated on solution quality, optimality gap, and time-to-solution.
- Hardware comparison: Consumer GPU (NVIDIA RTX 5070 Ti) vs. datacenter GPU (NVIDIA H100), assessing how GPU architecture affects cuOpt's scalability, memory limits, and solution quality across problem sizes ranging from tens to tens of thousands of nodes.

Both dimensions are evaluated using standard benchmark instances from the operations research literature, providing a comprehensive picture of where GPU-accelerated optimization delivers the most value and where its limits lie.

# Features
