from pulp import *
import pandas as pd

#Definir el problema
model=LpProblem("Minimizar_Distancia",LpMinimize)
datos=pd.read_csv("matriz_distancias.csv", index_col=0)
nodos=list(datos.index[:50])
datos=datos.loc[nodos, nodos]
root=nodos[0]
distancias={(i,j): datos.loc[i,j] for i in nodos for j in nodos if i!=j}

#Definir las variables de decisión
x=LpVariable.dicts("x",[(i,j) for i in nodos for j in nodos if i!=j],lowBound=0,upBound=1,cat='Binary')
u=LpVariable.dicts("u",nodos,lowBound=0,upBound=len(nodos)-1,cat='Integer')

#Definir la función objetivo
model+=lpSum(distancias[(i,j)]*x[(i,j)] for i in nodos for j in nodos if i!=j)

#Definrir las restricciones
for j in nodos:
	model+=lpSum(x[(i,j)] for i in nodos if i!=j)==1
for i in nodos:
	model+=lpSum(x[(i,j)] for j in nodos if i!=j)==1
for i in nodos:
	for j in nodos:
		if i!=j and i!=root and j!=root:
			model+=u[i]-u[j]+len(nodos)*x[(i,j)]<=len(nodos)-1
for i in nodos:
	if i!=root:
		model+=u[i]>=0
		model+=u[i]<=len(nodos)-1
model+=u[root]==0

#Resolver el problema
solver = PULP_CBC_CMD(msg=0)
model.solve(solver)
print("Status:", LpStatus[model.status])
print("Distancia total:", value(model.objective))