from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE, MINIMIZE, INTEGER
from cuopt.linear_programming.solver_settings import SolverSettings

#Definir problema
problem= Problem("Mochila")

#Definir datos
articulos=['articulo01','articulo02','articulo03','articulo04','articulo05','articulo06','articulo07','articulo08','articulo09','articulo10']
peso={'articulo01':50,'articulo02':25,'articulo03':30,'articulo04':40,'articulo05':80,'articulo06':60,'articulo07':45,'articulo08':10,'articulo09':20,'articulo10':90}
valor={'articulo01':501,'articulo02':278,'articulo03':318,'articulo04':470,'articulo05':673,'articulo06':671,'articulo07':528,'articulo08':117,'articulo09':197,'articulo10':1044}

#Definir variables
x={i: problem.addVariable(name=f"x_{i}", lb=0.0, ub=1.0, vtype=INTEGER) for i in articulos}

#Definir función objetivo
problem.setObjective(sum(valor[i]*x[i] for i in articulos), sense=MAXIMIZE)

#Definir restricciones
problem.addConstraint(sum(peso[i]*x[i] for i in articulos) <= 200, name="PesoMaximo")

#Resolver problema
problem.solve()
if problem.Status.name == "Optimal":
    print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
    print(f"Objective value = {problem.ObjValue}")
else:
    print(f"Problem status: {problem.Status.name}")