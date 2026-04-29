from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value

#Definir el problema de optimización
model=LpProblem("Minimizar_Costes", LpMinimize)
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

#Definir las variables de decisión
x=LpVariable.dicts("x", [(m,t) for m in transportes for t in parejas_ciudades], lowBound=0,upBound=1, cat='Binary')
y=LpVariable.dicts("y", [(m1,m2,t) for m1 in transportes for m2 in transportes for t in parejas_ciudades if m1!=m2], lowBound=0,upBound=1, cat='Binary')
r=LpVariable("r",lowBound=0, cat='Continuous')
d=LpVariable("d",lowBound=0, cat='Continuous')

#Definir la función objetivo
model += lpSum([costes_transporte[(m,t)]*x[(m,t)] for m in transportes for t in parejas_ciudades]) + lpSum([cambio_tansporte[(m1,m2)]*y[(m1,m2,t)] for m1 in transportes for m2 in transportes for t in parejas_ciudades if m1 != m2])+r*10

#Definir las restricciones
for t in parejas_ciudades:
    model += lpSum([x[(m,t)] for m in transportes]) == 1
for m1 in transportes:
    for m2 in transportes:
        for t in parejas_ciudades:
            if m1!=m2:
                siguiente=(t[1],t[1]+1)
                if siguiente in parejas_ciudades:
                    model += 1+y[(m1,m2,t)]>=x[(m1,t)]+x[(m2,siguiente)] 
model += d==lpSum([velocidad[(m,t)]*x[(m,t)] for m in transportes for t in parejas_ciudades]) + lpSum([tiempo_cambio[(m1,m2)]*y[(m1,m2,t)] for m1 in transportes for m2 in transportes for t in parejas_ciudades if m1 != m2])
model += r >= d-7

#Resolver el modelo
model.solve()
print('Estado:', LpStatus[model.status])
print(f'Coste total: {value(model.objective)}')

