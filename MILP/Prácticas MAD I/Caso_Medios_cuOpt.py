from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE, MINIMIZE, INTEGER
from cuopt.linear_programming.solver_settings import SolverSettings

#Definir problema
problem= Problem("Medios")

#Definir datos
transportes=['Ferrocarril','Carretera','Aire']
parejas_ciudades=[(1,2),(2,3),(3,4),(4,5)]
costes_transporte={('Ferrocarril',(1,2)):30, ('Ferrocarril',(2,3)):25, ('Ferrocarril',(3,4)):40, ('Ferrocarril',(4,5)):60,
                   ('Carretera',(1,2)):25, ('Carretera',(2,3)):40, ('Carretera',(3,4)):45, ('Carretera',(4,5)):50,
                   ('Aire',(1,2)):40, ('Aire',(2,3)):20, ('Aire',(3,4)):50, ('Aire',(4,5)):45}
cambio_tansporte={('Ferrocarril','Ferrocarril'):0, ('Ferrocarril','Carretera'):5, ('Ferrocarril','Aire'):12,
                    ('Carretera','Ferrocarril'):8, ('Carretera','Carretera'):0, ('Carretera','Aire'):10,
                    ('Aire','Ferrocarril'):15, ('Aire','Carretera'):10, ('Aire','Aire'):0}
velocidad={('Ferrocarril',(1,2)):3, ('Ferrocarril',(2,3)):7, ('Ferrocarril',(3,4)):4, ('Ferrocarril',(4,5)):4,
            ('Carretera',(1,2)):4, ('Carretera',(2,3)):9, ('Carretera',(3,4)):5, ('Carretera',(4,5)):6,
            ('Aire',(1,2)):1, ('Aire',(2,3)):1, ('Aire',(3,4)):1, ('Aire',(4,5)):1}
tiempo_cambio={('Ferrocarril','Ferrocarril'):0, ('Ferrocarril','Carretera'):0.5, ('Ferrocarril','Aire'):1,
                ('Carretera','Ferrocarril'):0.5, ('Carretera','Carretera'):0, ('Carretera','Aire'):0.5,
                ('Aire','Ferrocarril'):2, ('Aire','Carretera'):1, ('Aire','Aire'):0}

#Definir variables
x={(m,t): problem.addVariable(name=f"x_{m}_{t[0]}", lb=0.0, ub=1.0, vtype=INTEGER) for m in transportes for t in parejas_ciudades}
y={(m1,m2,t): problem.addVariable(name=f"y_{m1}_{m2}_{t[0]}", lb=0.0, ub=1.0, vtype=INTEGER) for m1 in transportes for m2 in transportes for t in parejas_ciudades if m1!=m2}
r=problem.addVariable(name="r", lb=0.0, vtype=CONTINUOUS)
d=problem.addVariable(name="d", lb=0.0, vtype=CONTINUOUS)

#Definir función objetivo
problem.setObjective(sum(costes_transporte[(m,t)]*x[(m,t)] for m in transportes for t in parejas_ciudades) + sum(cambio_tansporte[(m1,m2)]*y[(m1,m2,t)] for m1 in transportes for m2 in transportes for t in parejas_ciudades if m1 != m2) + r*10, sense=MINIMIZE)

#Definir restricciones
for t in parejas_ciudades:
    problem.addConstraint(sum(x[(m,t)] for m in transportes) == 1, name=f"Transporte_unico_{t}")
for m1 in transportes:
    for m2 in transportes:
        for t in parejas_ciudades:
            if m1!=m2:
                 siguiente=(t[1],t[1]+1)
                 if siguiente in parejas_ciudades:
                    problem.addConstraint(1+y[(m1,m2,t)]>=x[(m1,t)]+x[(m2,siguiente)], name=f"Transporte_cambio_{m1}_{m2}_{t}")

problem.addConstraint(d==sum(velocidad[(m,t)]*x[(m,t)] + sum(tiempo_cambio[(m1,m2)]*y[(m1,m2,t)] for m1 in transportes for m2 in transportes if m1 != m2) for m in transportes for t in parejas_ciudades), name="Definicion_tiempo_total")
problem.addConstraint(r >= d-7, name="Retraso")

#Resolver problema
problem.solve()
if problem.Status.name == "Optimal":
    print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
    print(f"Objective value = {problem.ObjValue}")
else:
    print(f"Problem status: {problem.Status.name}")