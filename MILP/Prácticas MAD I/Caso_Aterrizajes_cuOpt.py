from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE, MINIMIZE, INTEGER
from cuopt.linear_programming.solver_settings import SolverSettings

#Definir problema
problem= Problem("Aterrizajes")

#Definir datos
aviones=list(range(1,11))
F = [(i, j) for i in aviones for j in aviones if i != j]
E={1:129,2:195,3:89,4:96,5:110,6:120,7:124,8:126,9:135,10:160}
T={1:155,2:258,3:98,4:106,5:123,6:135,7:138,8:140,9:150,10:180}
L={1:559,2:744,3:510,4:521,5:555,6:576,7:577,8:573,9:591,10:657}
a={1:10,2:10,3:30,4:30,5:30,6:30,7:30,8:30,9:30,10:30}
b={1:10,2:10,3:30,4:30,5:30,6:30,7:30,8:30,9:30,10:30}
S={1:{1:0,2:3,3:15,4:15,5:15,6:15,7:15,8:15,9:15,10:15},
    2:{1:3,2:0,3:15,4:15,5:15,6:15,7:15,8:15,9:15,10:15},
    3:{1:15,2:15,3:0,4:8,5:8,6:8,7:8,8:8,9:8,10:8},
    4:{1:15,2:15,3:8,4:0,5:8,6:8,7:8,8:8,9:8,10:8},
    5:{1:15,2:15,3:8,4:8,5:0,6:8,7:8,8:8,9:8,10:8},
    6:{1:15,2:15,3:8,4:8,5:8,6:0,7:8,8:8,9:8,10:8},
    7:{1:15,2:15,3:8,4:8,5:8,6:8,7:0,8:8,9:8,10:8},
    8:{1:15,2:15,3:8,4:8,5:8,6:8,7:8,8:0,9:8,10:8},
    9:{1:15,2:15,3:8,4:8,5:8,6:8,7:8,8:8,9:0,10:8},
    10:{1:15,2:15,3:8,4:8,5:8,6:8,7:8,8:8,9:8,10:0}}
M=10000

#Definir variables
x={(i, j): problem.addVariable(name=f"x_{i}_{j}", lb=0.0, ub=1.0, vtype=INTEGER) for (i, j) in F}
t={i: problem.addVariable(name=f"t_{i}", lb=0.0, vtype=CONTINUOUS) for i in aviones}
e={i: problem.addVariable(name=f"e_{i}", lb=0.0, vtype=CONTINUOUS) for i in aviones}
d={i: problem.addVariable(name=f"d_{i}", lb=0.0, vtype=CONTINUOUS) for i in aviones}

#Definir función objetivo
problem.setObjective(sum(a[i]*e[i] + b[i]*d[i] for i in aviones), sense=MINIMIZE)

#Definir restricciones
for i in aviones:
    problem.addConstraint(t[i] >= E[i], name=f"Tiempo_minimo_{i}")
    problem.addConstraint(t[i] <= L[i], name=f"Tiempo_maximo_{i}")
    problem.addConstraint(d[i] >= t[i] - T[i], name=f"Retraso_{i}")
    problem.addConstraint(e[i] >= T[i] - t[i], name=f"Adelanto_{i}")
    problem.addConstraint(t[i] == T[i] + d[i] - e[i], name=f"Definicion_tiempo_{i}")
for (i, j) in F:
    problem.addConstraint(x[(i, j)] + x[(j, i)] == 1, name=f"Orden_{i}_{j}")
    problem.addConstraint(t[j] >= t[i] + S[i][j] - M + M * x[(i, j)], name=f"Separacion_{i}_{j}")

#Resolver problema
problem.solve()
if problem.Status.name == "Optimal":
    print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
    print(f"Objective value = {problem.ObjValue}")
else:
    print(f"Problem status: {problem.Status.name}")