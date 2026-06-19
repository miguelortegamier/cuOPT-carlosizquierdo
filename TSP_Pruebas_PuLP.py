from pulp import *
import pandas as pd
import math
import numpy as np

def calcular_distancia(a,b):
    dist=int(round(math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)))
    return dist

def get_subtours(x, locations):
    edges = [(i, j) for (i,j) in x if x[i,j].varValue > 0.5]
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
model= LpProblem("Minimizar_Distancia", LpMinimize)

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
x = {(i,j): LpVariable(name= f"x_{i}_{j}",cat='Binary') for i in locations for j in locations if i != j}

#Función objetivo
model += lpSum(cost_matrix[i][j] * x[i,j] for i in locations for j in locations if i != j)

#Restricciones
for j in locations:
    model += lpSum(x[i,j] for i in locations if i != j) == 1, f"Entrada_{j}"
for i in locations:
    model += lpSum(x[i,j] for j in locations if i != j) == 1, f"Salida_{i}"

#Loop DFJ
solver=CUOPT(msg=1, timeLimit=300)
iteracion = 0
tiempo_resolucion = 0.0
while True:
    model.solve(solver)
    tiempo_resolucion += model.solutionTime
    print(f"Iteración {iteracion}: Status = {model.status}, Objective = {value(model.objective):.2f}")
    subtours = get_subtours(x, locations)
    if len(subtours) == 1:
        print("Tour completo, terminado")
        break
    print(f" {len(subtours)} subtours encontrados, añadiendo cortes...")
    for S in subtours:
        model += lpSum(x[i,j] for i in S for j in S if i != j) <= len(S)-1, f"Subtour_{iteracion}_{S[0]}"
    iteracion += 1
print(f"\n Distancia Total: {value(model.objective):.2f}")
print(f"Tiempo Resolucion: {tiempo_resolucion:.2f} segundos")