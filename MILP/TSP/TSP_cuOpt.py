from cuopt.linear_programming.problem import *
from cuopt.linear_programming.solver_settings import SolverSettings
import pandas as pd
import numpy as np
import math

def calcular_distancia(a,b):
    dist=int(round(math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)))

    return dist

#Definir el problema
problem = Problem("TSP")

#Definir datos
datos=pd.read_csv("eil51.csv", sep=';')
datos=datos.head(40)
locations=list(range(len(datos)))
cost_matrix=np.zeros((len(locations), len(locations)))
for i in locations:
    for j in locations:
        punto_a=(datos.loc[i,"X_COORD"], datos.loc[i,"Y_COORD"])
        punto_b=(datos.loc[j,"X_COORD"], datos.loc[j,"Y_COORD"])
        distancia=calcular_distancia(punto_a, punto_b)
        cost_matrix[i][j]=distancia

#Definir variables de decisión
x = {(i,j): problem.addVariable(name= f"x_{i}_{j}", lb=0.0 , ub=1.0 , vtype=INTEGER) for i in locations for j in locations if i != j} 
u={ i: problem.addVariable(name= f"u_{i}", lb=0.0 , ub=(len(locations)-1), vtype=INTEGER) for i in locations}

#Función objetivo
problem.setObjective(sum(cost_matrix[i][j] * x[i,j] for i in locations for j in locations if i != j), sense=MINIMIZE)

#Restricciones
for j in locations:
    problem.addConstraint(sum(x[i,j] for i in locations if i != j) == 1, name=f"Entrada_{j}")
for i in locations:
    problem.addConstraint(sum(x[i,j] for j in locations if i != j) == 1, name=f"Salida_{i}")
for i in locations:
    for j in locations:
        if i != j and i >= 1 and j >= 1:
            problem.addConstraint(u[i] - u[j] + len(locations) * x[i,j] <= len(locations) - 1, name=f"Subtour_{i}_{j}")
problem.addConstraint(u[0] == 0, name="u_root")

#Resolver el problema
problem.solve()
if problem.Status.name == "Optimal":
    print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
    print(f"Objective value = {problem.ObjValue}")
else:
    print(f"Problem status: {problem.Status.name}")