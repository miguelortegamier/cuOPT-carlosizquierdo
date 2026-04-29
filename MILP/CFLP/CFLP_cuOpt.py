from cuopt.linear_programming.problem import *
from cuopt.linear_programming.solver_settings import SolverSettings
import pandas as pd
import numpy as np

#Definir el problema
problem = Problem("CFLP")

# Definir datos
datos = pd.read_csv("1Cap10.csv", sep=';')
locations= list(datos["Facility"].unique())
clientes = list(datos["Cliente"].unique())
capacidad= datos.loc[0, "Capacidad"]
coste_fijo= datos.loc[0, "Coste Fijo"]
matriz_costos = datos.pivot(index='Facility', columns='Cliente', values='Coste Transporte')
matriz_demanda= datos.pivot(index='Facility', columns='Cliente', values='Demanda')
pares_validos = set(zip(datos['Facility'], datos['Cliente']))

#Definir variables
x = {(i,j): problem.addVariable(name= f"x_{i}_{j}", lb=0.0 , ub=1.0 , vtype=INTEGER) for i in locations for j in clientes if (i,j) in pares_validos}
y = {i: problem.addVariable(name= f"y_{i}", lb=0.0 , ub=1.0 , vtype=INTEGER) for i in locations}

#Función objetivo
problem.setObjective(sum(coste_fijo * y[i] for i in locations)+sum(matriz_costos.loc[i, j] * x[i,j] for (i,j) in pares_validos), sense=MINIMIZE)

#Restricciones
for j in clientes:
    problem.addConstraint(sum(x[i,j] for i in locations if (i,j) in pares_validos) == 1, name=f"Cliente_{j}")
for i in locations:
    problem.addConstraint(sum(matriz_demanda.loc[i, j] * x[i,j] for j in clientes if (i,j) in pares_validos) <= capacidad * y[i], name=f"Capacidad_{i}")
 
#Resolver el problema
problem.solve()
if problem.Status.name == "Optimal":
    print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
    print(f"Objective value = {problem.ObjValue}")
else:
    print(f"Problem status: {problem.Status.name}")