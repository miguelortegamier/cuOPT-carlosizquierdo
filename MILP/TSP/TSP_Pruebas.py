from cuopt.linear_programming.problem import *
from cuopt.linear_programming.solver_settings import SolverSettings
import pandas as pd
import numpy as np
import math

def calcular_distancia(a,b):
    dist=int(round(math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)))
    return dist

def get_subtours(x, locations):
    edges = [(i, j) for (i,j) in x if x[i,j].getValue() > 0.5]
    siguiente = {i: j for i, j in edges}
    visited = set()
    subtours = []
    for start in locations:
        if start not in visited:
            subtour = []
            current = start
            while current not in visited:
                visited.add(current)
                subtour.append(current)
                current = siguiente.get(current,start)
            subtours.append(subtour)
    return subtours

#Definir el problema
problem = Problem("TSP")

#Definir datos
datos=pd.read_csv("pr76.csv", sep=';')
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

#Función objetivo
problem.setObjective(sum(cost_matrix[i][j] * x[i,j] for i in locations for j in locations if i != j), sense=MINIMIZE)

#Restricciones
for j in locations:
    problem.addConstraint(sum(x[i,j] for i in locations if i != j) == 1, name=f"Entrada_{j}")
for i in locations:
    problem.addConstraint(sum(x[i,j] for j in locations if i != j) == 1, name=f"Salida_{i}")

#Loop DFJ
settings = SolverSettings()
settings.set_parameter("time_limit", 300)
iteracion = 0
tiempo_resolucion = 0.0
while True:
    problem.solve(settings)
    tiempo_resolucion += problem.SolveTime
    print(f"Iteración {iteracion}: Status = {problem.Status.name}, Objective = {problem.ObjValue:.2f}")
    subtours = get_subtours(x, locations)
    if len(subtours) == 1:
        print("Tour completo, terminado")
        break
    print(f" {len(subtours)} subtours encontrados, añadiendo cortes...")
    for S in subtours:
        problem.addConstraint(sum(x[i,j] for i in S for j in S if i != j) <= len(S)-1, name=f"Subtour_{iteracion}_{S[0]}")
    iteracion += 1
print(f"\n Distancia Total: {problem.ObjValue:.2f}")
print(f"Tiempo Resolucion: {tiempo_resolucion:.2f} segundos")