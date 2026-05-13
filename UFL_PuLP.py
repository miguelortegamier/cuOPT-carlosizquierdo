from pulp import *

#Definir el problema de optimización
model=LpProblem("Minimizar_Costes",LpMinimize)
clientes=list(range(1,4))
intalaciones=list(range(1,4))
coste_fijo={1:1000,2:1500,3:1200}
coste_cliente_instalacion={(1,1):200,(1,2):300,(1,3):250, 
                            (2,1):400,(2,2):200,(2,3):300,
                            (3,1):300,(3,2):400,(3,3):200}

#Definir las variables de decisión
x=LpVariable.dicts("x",[(i,j) for i in clientes for j in intalaciones],lowBound=0,upBound=1,cat='Binary')
y=LpVariable.dicts("y",[j for j in intalaciones],lowBound=0,upBound=1,cat='Binary')

#Definir la función objetivo
model+=lpSum(coste_fijo[j]*y[j] for j in intalaciones) + lpSum(coste_cliente_instalacion[(i,j)]*x[(i,j)] for i in clientes for j in intalaciones)

#Definir las restricciones
for i in clientes:
    model+=lpSum(x[(i,j)] for j in intalaciones)==1
for i in clientes:
    for j in intalaciones:
        model+=x[(i,j)]<=y[j]

#Resolver el problema
solver=CUOPT(msg=1, timeLimit=3000)
model.solve(solver)
print("Status:", LpStatus[model.status])
print("Coste total:", value(model.objective))