from pulp import *
import pandas as pd
import math
import numpy as np

def calcular_distancia(a,b):
    dist=int(round(math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)))

    return dist

# Definir el modelo
model = LpProblem("Minimizar_Distancia", LpMinimize)
datos = pd.read_csv("eil51.csv", sep=';')

# Cargar información de los nodos (nombres y ubicaciones)
locations=list(range(len(datos)))
cost_matrix=np.zeros((len(locations), len(locations)))
for i in locations:
    for j in locations:
        punto_a=(datos.loc[i,"X_COORD"], datos.loc[i,"Y_COORD"])
        punto_b=(datos.loc[j,"X_COORD"], datos.loc[j,"Y_COORD"])
        distancia=calcular_distancia(punto_a, punto_b)
        cost_matrix[i][j]=distancia

# Definir las variables de decisión
x = {(i, j): LpVariable(f"x_{i}_{j}", cat='Binary') for i in locations for j in locations if i != j}
u = {i: LpVariable(f"u_{i}", lowBound=0, upBound=(len(locations) - 1), cat='Integer') for i in locations}

# Función objetivo
model += lpSum([cost_matrix[i][j] * x[(i, j)] for i in locations for j in locations if i != j])

# Restricciones
for j in locations:
    model += lpSum([x[(i, j)] for i in locations if i != j]) == 1

for i in locations:
    model += lpSum([x[(i, j)] for j in locations if i != j]) == 1

for i in locations:
    for j in locations:
        if i != j and i >= 1 and j >= 1:
            model += u[i] - u[j] + len(locations) * x[(i, j)] <= len(locations) - 1

model += u[0] == 0

# Resolver y mostrar resultados 
print("Resolviendo con CUOPT...")
solver=PULP_CBC_CMD()
model.solve(solver)
print('Estado:', LpStatus[model.status])
print('Distancia total (mejor solución encontrada):', value(model.objective))