from cuopt.linear_programming.problem import *
from cuopt.linear_programming.solver_settings import SolverSettings
from pulp import *
import pandas as pd
import numpy as np
import math
import glob
import os

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

def leer_solucion(nombre_archivo):
    with open(nombre_archivo, "r", encoding="utf-8") as f:
        funcion_objetivo = f.read().split()[-1]
        funcion_objetivo = float(funcion_objetivo)
    return funcion_objetivo

archivos = sorted(archivo for archivo in glob.glob('Euclid/*.txt') if os.path.exists(f"{archivo}.opt"))

#Definir datos
for archivo in archivos:
    optimo=leer_solucion(f"{archivo}.opt")
    costes_variables, coste_fijo, clientes, instalaciones = leer_matriz(archivo)
    clientes=list(range(clientes))
    instalaciones=list(range(instalaciones))
    coste_cliente_instalacion=costes_variables
    coste_fijo={j: coste_fijo[j] for j in instalaciones}

    #CuOpt API
    problem=Problem("UFL")
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
    gap = float("nan")
    if problem.Status.name == "Optimal":
        print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
        print(f"Objective value = {problem.ObjValue}")
        gap= abs((problem.ObjValue - optimo)) / optimo * 100
    else:
        print(f"Problem status: {problem.Status.name}")
    with open("resultados_cuOpt1.txt", "a") as f:
        f.write(f"Archivo: {archivo}, Solver: cuOpt_API, Status: {problem.Status.name}, Objective value: {problem.ObjValue}, Gap: {gap:.2f}%, Solve time: {problem.SolveTime:.2f} seconds\n")
    
    #PuLP
    model = LpProblem("UFL", LpMinimize)
    #Definir variables de decisión
    x= {(i,j): LpVariable(name=f"x_{i}_{j}", lowBound=0, upBound=1, cat='Binary') for i in clientes for j in instalaciones}
    y= {j: LpVariable(name=f"y_{j}", cat='Binary') for j in instalaciones}
    #Definir la función objetivo
    model += lpSum(coste_fijo[j]*y[j] for j in instalaciones) + lpSum(coste_cliente_instalacion[(i,j)]*x[(i,j)] for i in clientes for j in instalaciones)
    #Definir las restricciones
    for i in clientes:
        model += lpSum(x[(i,j)] for j in instalaciones) == 1, f"Asignacion_cliente_{i}"
        for j in instalaciones:
            model += x[(i,j)] <= y[j], f"Restriccion_cliente_{i}_instalacion_{j}"
    # Resolver el problema con varios solvers de PuLP
    solvers=[CUOPT(msg=1, timeLimit=300),PULP_CBC_CMD(msg=1, timeLimit=300)]
    for solver in solvers:
        model.solve(solver)
        if model.status == LpStatusOptimal:
            print(f"[{solver.__class__.__name__}] Optimal solution found in {model.solutionTime:.2f} seconds")
            print(f"[{solver.__class__.__name__}] Objective value = {value(model.objective)}")
            gap= abs((value(model.objective) - optimo) / optimo) * 100
        else:
            print(f"[{solver.__class__.__name__}] Problem status: {model.status}")
        with open("resultados_cuOpt1.txt", "a") as f:
            f.write(f"Archivo: {archivo}, Solver: {solver.__class__.__name__}, Status: {model.status}, "f"Objective value: {value(model.objective)}, Gap: {gap:.2f}%, Solve time: {model.solutionTime:.2f} seconds\n")

# Resumen Mejor Solver
resultados = {}
for line in open("resultados_cuOpt1.txt"):
    archivo = line.split("Archivo: ")[1].split(",")[0]
    solver = line.split("Solver: ")[1].split(",")[0]
    gap = float(line.split("Gap: ")[1].split("%")[0])
    tiempo = float(line.split("Solve time: ")[1].split()[0])
    resultados.setdefault(archivo, []).append((solver, gap, tiempo))

with open("resumen_mejor_solver.txt", "w") as f:
    for archivo in sorted(resultados):
        mejor_gap = min(resultados[archivo], key=lambda x: x[1])
        mejor_tiempo = min(resultados[archivo], key=lambda x: x[2])
        f.write(f"{archivo}: {mejor_gap[0]} ({mejor_gap[1]:.2f}%), {mejor_tiempo[0]} ({mejor_tiempo[2]:.2f}s)\n")
