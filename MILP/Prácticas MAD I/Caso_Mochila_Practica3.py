from pulp import LpProblem, LpMaximize, LpVariable, lpSum,CUOPT
# Crear el modelo de optimizacion
model=LpProblem(name="Mochila", sense=LpMaximize)
articulos=['articulo01','articulo02','articulo03','articulo04','articulo05','articulo06','articulo07','articulo08','articulo09','articulo10']
peso={'articulo01':50,'articulo02':25,'articulo03':30,'articulo04':40,'articulo05':80,'articulo06':60,'articulo07':45,'articulo08':10,'articulo09':20,'articulo10':90}
valor={'articulo01':501,'articulo02':278,'articulo03':318,'articulo04':470,'articulo05':673,'articulo06':671,'articulo07':528,'articulo08':117,'articulo09':197,'articulo10':1044}

# Definir las variables de decision
x=LpVariable.dicts('x',articulos, lowBound=0,cat='Binary')

#Definir la funcion objetivo
model+= lpSum([valor[i]*x[i] for i in articulos])

#Definir las restricciones
model+= lpSum([peso[i]*x[i] for i in articulos]) <= 200

#Resolver el modelo
model.solve(CUOPT())
for i in articulos:
    if x[i].varValue==1:
        print(f'Seleccionado: {i}')
print(f'Valor total de la mochila: {model.objective.value()}')
