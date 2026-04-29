from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE, MINIMIZE, INTEGER
from cuopt.linear_programming.solver_settings import SolverSettings

#Definir problema
problem= Problem("Localizacion")

#Definir datos
cliente = list(range(1, 7))
demanda = {1:120,2:80,3:75,4:100,5:110,6:100}
almacen = list(range(1, 7))
coste = {1:3.5,2:9,3:10,4:4,5:3,6:3}
capacidad = {1:400,2:350,3:200,4:280,5:375,6:400}
coste_envio = {1:{1:100,2:80,3:50,4:50,5:60,6:100},
             2:{1:120,2:90,3:60,4:70,5:65,6:110},
             3:{1:140,2:110,3:80,4:80,5:75,6:130},
             4:{1:160,2:125,3:100,4:100,5:80,6:150},
             5:{1:190,2:150,3:130,4:None,5:None,6:None},
             6:{1:200,2:50,3:150,4:None,5:None,6:None}}
posibleatender = {(a, c) for a in almacen for c in cliente if coste_envio[a].get(c) is not None}

#Definir variables
x={a: problem.addVariable(name=f"x_{a}", lb=0.0, ub=1.0, vtype=INTEGER) for a in almacen}
y={(a, c): problem.addVariable(name=f"y_{a}_{c}", lb=0.0, ub=1.0, vtype=CONTINUOUS) for (a,c) in posibleatender}

#Definir función objetivo
problem.setObjective(sum(coste[a]*x[a] for a in almacen)+sum(coste_envio[a][c]*y[(a, c)] for (a, c) in posibleatender), sense=MINIMIZE)

#Definir restricciones
for c in cliente:
    problem.addConstraint(sum(y[(a, c)] for a in almacen if (a, c) in posibleatender) == 1, name=f"Demanda_{c}")
for a in almacen:
    problem.addConstraint(sum(demanda[c]*y[(a, c)] for c in cliente if (a, c) in posibleatender) <= capacidad[a], name=f"Capacidad_{a}")
for a in almacen:
    for c in cliente:
        if (a, c) in posibleatender:
            problem.addConstraint(y[(a, c)] <= x[a], name=f"Atencion_{a}_{c}")
problem.addConstraint(sum(x[a] for a in almacen) <= 2, name="MaxAlmacenes")

#Resolver problema
problem.solve()
if problem.Status.name == "Optimal":
    print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
    print(f"Objective value = {problem.ObjValue}")
else:
    print(f"Problem status: {problem.Status.name}")