from pulp import *
import pandas as pd
import math
import numpy as np

# Definir el problema
model = LpProblem("CFLP", LpMinimize)

#Definir datos
datos = pd.read_csv("1Cap10.csv", sep=';')
locations= list(datos["Facility"].unique())
clientes = list(datos["Cliente"].unique())
capacidad= datos.loc[0, "Capacidad"]
coste_fijo= datos.loc[0, "Coste Fijo"]
matriz_costos = datos.pivot(index='Facility', columns='Cliente', values='Coste Transporte')
matriz_demanda= datos.pivot(index='Facility', columns='Cliente', values='Demanda')
pares_validos = set(zip(datos['Facility'], datos['Cliente']))

#Definir variables
x = {(i,j): LpVariable(f"x_{i}_{j}", cat='Binary') for i in locations for j in clientes if (i,j) in pares_validos}
y= {i: LpVariable(f"y_{i}", cat='Binary') for i in locations}

#Definir función objetivo
model += lpSum(coste_fijo * y[i] for i in locations) + lpSum(matriz_costos.loc[i, j] * x[i,j] for (i,j) in pares_validos)

#Definir restricciones
for j in clientes:
    model += lpSum(x[i,j] for i in locations if (i,j) in pares_validos) == 1, f"Cliente_{j}"
for i in locations:
    model += lpSum(matriz_demanda.loc[i, j] * x[i,j] for j in clientes if (i,j) in pares_validos) <= capacidad * y[i], f"Capacidad_{i}"

#Resolver el problema
print("Resolviendo con CUOPT...")
solver=CUOPT()
model.solve(solver)
print('Estado:', LpStatus[model.status])
print('Distancia total (mejor solución encontrada):', value(model.objective))