from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, CUOPT

#Definir el modelo
model=LpProblem("Minimizar_Multas", LpMinimize)
aviones=list(range(1,11))
F = [(i, j) for i in aviones for j in aviones if i != j]
llegada_mas_temprana={1:129,2:195,3:89,4:96,5:110,6:120,7:124,8:126,9:135,10:160}
hora_objetivo={1:155,2:258,3:98,4:106,5:123,6:135,7:138,8:140,9:150,10:180}
llegada_mas_tardia={1:559,2:744,3:510,4:521,5:555,6:576,7:577,8:573,9:591,10:657}
multa_adelanto={1:10,2:10,3:30,4:30,5:30,6:30,7:30,8:30,9:30,10:30}
multa_retraso={1:10,2:10,3:30,4:30,5:30,6:30,7:30,8:30,9:30,10:30}
intervalo_aterrizajes={1:{1:0,2:3,3:15,4:15,5:15,6:15,7:15,8:15,9:15,10:15},
                       2:{1:3,2:0,3:15,4:15,5:15,6:15,7:15,8:15,9:15,10:15},
                        3:{1:15,2:15,3:0,4:8,5:8,6:8,7:8,8:8,9:8,10:8},
                        4:{1:15,2:15,3:8,4:0,5:8,6:8,7:8,8:8,9:8,10:8},
                        5:{1:15,2:15,3:8,4:8,5:0,6:8,7:8,8:8,9:8,10:8},
                        6:{1:15,2:15,3:8,4:8,5:8,6:0,7:8,8:8,9:8,10:8},
                        7:{1:15,2:15,3:8,4:8,5:8,6:8,7:0,8:8,9:8,10:8},
                        8:{1:15,2:15,3:8,4:8,5:8,6:8,7:8,8:0,9:8,10:8},
                        9:{1:15,2:15,3:8,4:8,5:8,6:8,7:8,8:8,9:0,10:8},
                        10:{1:15,2:15,3:8,4:8,5:8,6:8,7:8,8:8,9:8,10:0}}

#Definir las variables de decisión
x=LpVariable.dicts("x", ((i,j) for i in aviones for j in aviones if i!=j), lowBound=0, upBound=1, cat='Binary')
t=LpVariable.dicts("t", aviones, lowBound=0, cat='Integer')
e=LpVariable.dicts("e", aviones, lowBound=0, cat='Integer')
d=LpVariable.dicts("d", aviones, lowBound=0, cat='Integer')

#Definir la función objetivo
model += lpSum([multa_adelanto[i]*e[i] + multa_retraso[i]*d[i] for i in aviones])

#Definir las restricciones
for i in aviones:
    model += llegada_mas_temprana[i] <= t[i] <= llegada_mas_tardia[i]
    model += d[i] >= t[i]-hora_objetivo[i]
    model += e[i] >= hora_objetivo[i]-t[i]
    model += t[i] == hora_objetivo[i] + d[i] - e[i]
for (i, j) in F:
    model += x[(i, j)] + x[(j, i)] == 1
    model += t[j] >= t[i] + intervalo_aterrizajes[i][j] - 1000000 * (1 - x[(i, j)])
        
#Resolver el modelo
model.solve(CUOPT())
print('Estado:', LpStatus[model.status])
print(f'Multa total: {value(model.objective)}')