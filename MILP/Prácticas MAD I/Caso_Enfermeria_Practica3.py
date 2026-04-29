from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, CUOPT, PULP_CBC_CMD

# Crear el modelo de optimización
model = LpProblem("Minimizar_Tamaño_Plantilla", LpMinimize)
numero = list(range(12))
trabajadores = {0:15,1:15,2:15,3:35,4:40,5:40,6:40,7:30,8:31,9:35,10:30,11:20}

# Definir las variables de decisión (enteras)
x = LpVariable.dicts("x", numero, lowBound=0, cat='Integer')
y = LpVariable.dicts("y", numero, lowBound=0, cat='Integer')

# Definir la función objetivo
model += lpSum([y[i] for i in numero])

# Definir las restricciones
# Se usan índices mod 12 para manejar periodos anteriores (rotación)
for i in numero:
    model += ( x[i] + x[(i-1) % 12] + x[(i-3) % 12] + x[(i-4) % 12] + y[(i-5) % 12] >= trabajadores[i])
    model += y[i] <= x[i]

model += lpSum([x[i] for i in numero]) <= 80

# Resolver el modelo
model.solve(CUOPT(mip=True))
print('Estado:', LpStatus[model.status])
print(f'Número total de horas extra: {value(model.objective)}')
for i in numero:
    print(f'Periodo {i}: x={x[i].varValue}, y={y[i].varValue}')