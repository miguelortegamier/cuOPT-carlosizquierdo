from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE, MINIMIZE, INTEGER
from cuopt.linear_programming.solver_settings import SolverSettings

#Definir el problema
problem = Problem("Minas")

#Definir datos
niveles=list(range(1,4))
columnas=list(range(1,9))
niveles_convencional=list(range(1,3))
niveles_nueva=list(range(2,4))
columnas_nivel={1:[1,2,3,4,5,6,7,8],2:[2,3,4,5,6,7],3:[3,4,5,6]}
beneficios={(1,1):200,(1,2):0,(1,3):0,(1,4):0,(1,5):0,(1,6):0,(1,7):300,(1,8):0,
    (2,2):0,(2,3):500,(2,4):0,(2,5):200,(2,6):0,(2,7):0,
    (3,3):0,(3,4):0,(3,5):1000,(3,6):1200}
costes={(1,1):100,(1,2):100,(1,3):100,(1,4):100,(1,5):100,(1,6):100,(1,7):100,(1,8):100,
(2,2):1000,(2,3):200,(2,4):200,(2,5):200,(2,6):200,(2,7):1000,
(3,3):1000,(3,4):1000,(3,5):300,(3,6):1000}
toneladas={(1,1):10000,(1,2):10000,(1,3):10000,(1,4):10000,(1,5):10000,(1,6):10000,(1,7):10000,(1,8):10000,
    (2,2):10000,(2,3):10000,(2,4):10000,(2,5):10000,(2,6):10000,(2,7):10000,
    (3,3):10000,(3,4):10000,(3,5):10000,(3,6):10000}

#Definir variables
x={(n,c): problem.addVariable(name=f"x_{n}_{c}", lb=0.0, ub=1.0, vtype=INTEGER) for n in niveles for c in columnas}

#Definir función objetivo
problem.setObjective(sum([(beneficios[(n,c)]-costes[(n,c)]*1.25)*toneladas[(n,c)]*x[(n,c)] for n in niveles for c in columnas_nivel[n]]), sense=MAXIMIZE)

#Definir restricciones
for n in niveles_convencional:
    for c in columnas_nivel[n]:
        if n>1:
            problem.addConstraint(3*x[(n,c)] <= x[(n-1,c-1)] + x[(n-1,c)] + x[(n-1,c+1)], name=f"Convencional_{n}_{c}")
for n in niveles_nueva:
    for c in columnas_nivel[n]:
        problem.addConstraint(x[(n,c)] <= x[(n-1,c)], name=f"Nueva_{n}_{c}")
    
#Resolver problema
problem.solve()
if problem.Status.name == "Optimal":
    print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
    print(f"Objective value = {problem.ObjValue}")
else:
    print(f"Problem status: {problem.Status.name}")