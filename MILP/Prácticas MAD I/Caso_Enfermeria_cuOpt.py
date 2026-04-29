from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE, MINIMIZE, INTEGER
from cuopt.linear_programming.solver_settings import SolverSettings

#Definir problema
problem= Problem("Enfermeria")

#Definir datos
numero = list(range(12))
trabajadores = {0:15,1:15,2:15,3:35,4:40,5:40,6:40,7:30,8:31,9:35,10:30,11:20}

#Definir variables
x={(i): problem.addVariable(name=f"x_{i}", lb=0.0, vtype=INTEGER) for i in numero}
y={(i): problem.addVariable(name=f"y_{i}", lb=0.0, vtype=INTEGER) for i in numero}

#Definir función objetivo
problem.setObjective(sum(y[i] for i in numero), sense=MINIMIZE)

#Definir restricciones
for i in numero:
    problem.addConstraint(x[i] + x[(i-1) % 12] + x[(i-3) % 12] + x[(i-4) % 12] + y[(i-5) % 12] >= trabajadores[i], name=f"Trabajadores_{i}")
problem.addConstraint(sum(x[i] for i in numero) <= 80, name="Maximo_trabajadores")

#Resolver problema
problem.solve()
if problem.Status.name == "Optimal":
    print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
    print(f"Objective value = {problem.ObjValue}")
else:
    print(f"Problem status: {problem.Status.name}")