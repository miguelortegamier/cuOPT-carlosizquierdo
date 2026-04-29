from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value,CUOPT

# Crear el modelo de optimización
model = LpProblem("Minimizar_Costes", LpMinimize)

cliente = list(range(1, 13))
demanda = {1:120,2:80,3:75,4:100,5:110,6:100,7:90,8:60,9:30,10:150,11:95,12:120}
almacen = list(range(1, 13))
coste = {1:3.5,2:9,3:10,4:4,5:3,6:9,7:9,8:3,9:4,10:10,11:9,12:3.5}
capacidad = {1:300,2:250,3:100,4:180,5:275,6:300,7:200,8:220,9:270,10:250,11:230,12:180}
coste_envio = {1:{1:100,2:80,3:50,4:50,5:60,6:100,7:120,8:90,9:60,10:70,11:65,12:110},
             2:{1:120,2:90,3:60,4:70,5:65,6:110,7:140,8:110,9:80,10:80,11:75,12:130},
             3:{1:140,2:110,3:80,4:80,5:75,6:130,7:160,8:125,9:100,10:100,11:80,12:150},
             4:{1:160,2:125,3:100,4:100,5:80,6:150,7:190,8:150,9:130,10:None,11:None,12:None},
             5:{1:190,2:150,3:130,4:None,5:None,6:None,7:200,8:180,9:150,10:None,11:None,12:None},
             6:{1:200,2:180,3:150,4:None,5:None,6:None,7:100,8:80,9:50,10:50,11:60,12:100},
             7:{1:100,2:80,3:50,4:50,5:60,6:100,7:120,8:90,9:60,10:70,11:65,12:110},
             8:{1:120,2:90,3:60,4:70,5:65,6:110,7:140,8:110,9:80,10:80,11:75,12:130},
             9:{1:140,2:110,3:80,4:80,5:75,6:130,7:160,8:125,9:100,10:100,11:80,12:150},
             10:{1:160,2:125,3:100,4:100,5:80,6:150,7:190,8:150,9:130,10:None,11:None,12:None},
             11:{1:190,2:150,3:130,4:None,5:None,6:None,7:200,8:180,9:150,10:None,11:None,12:None},
             12:{1:200,2:180,3:150,4:None,5:None,6:None,7:100,8:80,9:50,10:50,11:80,12:100}}

# Construir parámetro posible_atender y limpiar coste_envio (None -> 0.0)
posible_atender_param = {}
coste_envio_clean = {i: {} for i in almacen}
for i in almacen:
    for j in cliente:
        if coste_envio[i].get(j) is None:
            posible_atender_param[(i, j)] = 0
            coste_envio_clean[i][j] = 0.0
        else:
            posible_atender_param[(i, j)] = 1
            coste_envio_clean[i][j] = float(coste_envio[i][j])

# Definir las variables de decisión
x = LpVariable.dicts("x", almacen, lowBound=0, upBound=1, cat='Binary')
y = LpVariable.dicts("y", [(i, j) for i in almacen for j in cliente], lowBound=0, upBound=1, cat='Continuous')

# Definir la función objetivo
model += (lpSum(coste[i] * x[i] for i in almacen) + lpSum(coste_envio_clean[i][j] * y[(i, j)] for i in almacen for j in cliente))*1000

# Definir las restricciones
for j in cliente:
    model += lpSum(y[(i, j)] for i in almacen) == 1

for i in almacen:
    model += lpSum(demanda[j] * y[(i, j)] for j in cliente) <= capacidad[i]

for i in almacen:
    for j in cliente:
        model += y[(i, j)] <= posible_atender_param[(i, j)] * x[i]

model += lpSum(x[i] for i in almacen) <= 4
# Resolver el modelo
model.solve(CUOPT())
print('Estado:', LpStatus[model.status])
print(f'Coste total: {value(model.objective)}')
