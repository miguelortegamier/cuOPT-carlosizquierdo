from cuopt.linear_programming.problem import *
from cuopt.linear_programming.solver_settings import SolverSettings
import pandas as pd
import numpy as np
import math

def leer_matriz(nombre_archivo):
    with open(nombre_archivo, "r", encoding="utf-8") as f:
        f.readline()  # Saltar cabecera: "FILE: ..."
        todo = np.fromstring(f.read(), sep=" ")
    clientes = int(todo[0])   
    instalaciones = int(todo[1]) 
    datos= todo[3:]
    ancho= instalaciones + 2
    matriz= datos.reshape((clientes, ancho))
    costes_variables = matriz[:, 2:]
    coste_fijo=matriz[:, 1]

    return costes_variables, coste_fijo, clientes, instalaciones

# Definir el problema
problem=Problem("UFL")

#Definir datos
costes_variables, coste_fijo, clientes, instalaciones = leer_matriz('Euclid/111EuclS.txt')
clientes=list(range(clientes))
instalaciones=list(range(instalaciones))
coste_cliente_instalacion=costes_variables
coste_fijo={j: coste_fijo[j] for j in instalaciones}

#Definir variables de decisión
x={(i,j):problem.addVariable(name=f"x_{i}_{j}",lb=0, ub=1, vtype=INTEGER) for i in clientes for j in instalaciones}
y={j:problem.addVariable(name=f"y_{j}",lb=0, ub=1, vtype=INTEGER) for j in instalaciones}

#Definir la función objetivo
problem.setObjective(sum(coste_fijo[j]*y[j] for j in instalaciones) + sum(coste_cliente_instalacion[(i,j)]*x[(i,j)] for i in clientes for j in instalaciones), sense="minimize")

#Definir las restricciones
for i in clientes:
    problem.addConstraint(sum(x[(i,j)] for j in instalaciones) == 1, name=f"Asignacion_cliente_{i}")
    for j in instalaciones:
        problem.addConstraint(x[(i,j)] <= y[j], name=f"Restriccion_cliente_{i}_instalacion_{j}")

#Resolver el problema
settings = SolverSettings()
settings.set_parameter("time_limit", 300)
problem.solve(settings)
if problem.Status.name == "Optimal":
    print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
    print(f"Objective value = {problem.ObjValue}")
else:
    print(f"Problem status: {problem.Status.name}")