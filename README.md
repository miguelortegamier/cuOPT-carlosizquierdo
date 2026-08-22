# NVIDIA cuOpt
Carlos Izquierdo Hernández's research fellowship project at the Department of Industrial Management, Business Administration and Statistics of Universidad Politécnica de Madrid (UPM).<br> 
___
*"What if a consumer GPU could match a commercial solver in logistics?" — CIO 2026, Alcoy, 2–3 July 2026*
# Overview
This project develops a comparative study applied to Mixed Integer Linear Programming (MILP) and Vehicle Routing Problems (VRP):
- **Solver comparison:** Traditional CPU-based solvers (CBC and Gurobi) vs. NVIDIA cuOpt evaluated on solution quality, optimality gap, and time-to-solution across four interfaces: CBC (PuLP), Gurobi (PuLP), cuOpt (PuLP), and cuOpt (native API).
- **Hardware comparison:** Consumer GPU (NVIDIA RTX 5070 Ti) vs. datacenter GPU (NVIDIA H100), assessing scalability limits, memory constraints, and solution quality across problem sizes ranging from tens to 30,000+ nodes.

Both dimensions are evaluated using standard benchmark instances from the operations research literature, providing a comprehensive picture of where GPU-accelerated optimization delivers the most value and where its limits lie. <br>
___
The research was presented at **CIO 2026** (Congreso de Ingeniería de Organización), Alcoy, 2–3 July 2026. <br>
**Authors:** *Carlos Izquierdo Hernández · Miguel Ortega Mier* 

# Features
- **Multi-Solver Comparison:** Systematic benchmarking of CBC, Gurobi, and NVIDIA cuOpt across identical problem instances, measuring time-to-solution and optimality gap.
- **GPU vs. GPU Hardware Benchmarking:** Head-to-head comparison of a consumer GPU (RTX 5070 Ti) against a professional datacenter GPU (H100), identifying the scalability ceiling of each and the practical trade-offs between cost and performance.
- **Broad Problem Coverage:** Includes classical MILP problems (knapsack, facility location, scheduling) and routing problems (TSP, CVRP, VRPTW, heterogeneous fleet VRP).
- **Large-Scale Benchmarking:** Evaluated against Taillard, Christofides & Eilon, Golden, Gehring & Homberger (VRPTW), and Arnold, Gendreau & Sörensen instances (up to 30,001 nodes).
- **MILP Benchmarking:** TSPlib and UFLlib instances solved with Gurobi, CBC, cuOpt (PuLP) and cuOpt (API), with identical solution quality across all four solvers.
- **PuLP Integration:** All MILP models defined using PuLP's Python API, enabling clean formulation before delegating computation to cuOpt or classical solvers.
