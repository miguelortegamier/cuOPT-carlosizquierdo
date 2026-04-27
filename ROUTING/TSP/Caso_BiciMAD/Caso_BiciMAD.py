from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD, GUROBI_CMD
import pandas as pd
import time
import folium #type: ignore
from folium import plugins # type: ignore
import webbrowser
import os

# Definir el modelo
model = LpProblem("Minimizar_Distancia", LpMinimize)
datos = pd.read_csv("tsp_madrid_81_dist_matrix_real_road.csv", index_col=0)

# Cargar información de los nodos (nombres y ubicaciones)
nodos_info = pd.read_csv("tsp_madrid_81_nodes_real.csv")
nombres_lugares = {row['node_id']: row['nombre'] for _, row in nodos_info.iterrows()}
datos.columns = datos.columns.astype(int)
nodos = list(datos.index)
datos = datos.loc[nodos, nodos]
N = len(nodos)
root = nodos[0]
nodo_a_num = {nodo: i for i, nodo in enumerate(nodos)}
num_a_nodo = {i: nodo for i, nodo in enumerate(nodos)}
distancias = {(i, j): datos.loc[i, j] for i in nodos for j in nodos if i != j}

# Definir las variables de decisión
x = {(i, j): LpVariable(f"x_{i}_{j}", cat='Binary') for i in nodos for j in nodos if i != j}
u = {i: LpVariable(f"u_{i}", lowBound=0, upBound=N - 1, cat='Integer') for i in nodos}

# Función objetivo
model += lpSum([distancias[(i, j)] * x[(i, j)] for i in nodos for j in nodos if i != j])

# Restricciones
for j in nodos:
    model += lpSum([x[(i, j)] for i in nodos if i != j]) == 1

for i in nodos:
    model += lpSum([x[(i, j)] for j in nodos if i != j]) == 1

for i in nodos:
    for j in nodos:
        if i != j and nodo_a_num[i]>=1 and nodo_a_num[i]<=N and nodo_a_num[j]>=1 and nodo_a_num[j]<=N:
            model += u[i] - u[j] + N * x[(i, j)] <= N - 1

for i in nodos:
    if nodo_a_num[i]>=1 and nodo_a_num[i]<=N:
        model += u[i] >= 0
        model += u[i] <= N - 1

model += u[root] == 0

# Resolver y mostrar resultados 
print("Resolviendo con Gurobi...")
solver=GUROBI_CMD(msg=1, timeLimit=120)
model.solve(solver)
print('Estado:', LpStatus[model.status])
print('Distancia total (mejor solución encontrada):', value(model.objective))

# Creado con IA para mostrar el recorrido completo y generar un mapa interactivo con la ruta del TSP.
if model.status in [1, -1]:  # Optimal o encontró alguna solución
    # Crear diccionario de arcos
    siguiente = {}
    for i in nodos:
        for j in nodos:
            if i != j and x[(i, j)].varValue is not None and x[(i, j)].varValue == 1:
                origen_nombre = nombres_lugares.get(i, f"Nodo {i}")
                destino_nombre = nombres_lugares.get(j, f"Nodo {j}")
                siguiente[i] = j
    
    # Reconstruir y mostrar el recorrido completo
    print('\n' + '='*80)
    print('RECORRIDO COMPLETO DEL CICLO:')
    print('='*80)
    actual = root
    ruta = [actual]
    distancia_acumulada = 0
    
    for _ in range(len(nodos)):
        if actual in siguiente:
            anterior = actual
            actual = siguiente[actual]
            ruta.append(actual)
            if actual == root:
                break
    
    # Mostrar ruta detallada con nombres y distancias
    print(f"\nInicio: {nombres_lugares.get(ruta[0], f'Nodo {ruta[0]}')}")
    print('-' * 80)
    
    for idx in range(1, len(ruta)):
        origen = ruta[idx-1]
        destino = ruta[idx]
        distancia = distancias.get((origen, destino), 0)
        distancia_acumulada += distancia
        
        nombre_destino = nombres_lugares.get(destino, f"Nodo {destino}")
        
        if destino == root:
            print(f"\nRetorno a: {nombre_destino}")
            print(f"   Distancia desde ubicación anterior: {round(distancia, 2)} km")
        else:
            print(f"{idx}. {nombre_destino}")
            print(f"   Distancia desde ubicación anterior: {round(distancia, 2)} km")
            print(f"   Distancia acumulada: {round(distancia_acumulada, 2)} km")
            print()
    
    print('\n' + '='*80)
    print(f'Total de ubicaciones visitadas: {len(ruta)-1}')
    print(f'Distancia total del recorrido: {round(value(model.objective), 2)} km')
    print('='*80)
    
    # ===== VISUALIZACIÓN EN MAPA =====
    print('\nGenerando mapa...')
    
    # Obtener coordenadas
    coordenadas = {row['node_id']: (row['lat'], row['lon']) for _, row in nodos_info.iterrows()}
    
    # Crear el mapa centrado en Madrid
    centro_lat = nodos_info['lat'].mean()
    centro_lon = nodos_info['lon'].mean()
    mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=13, tiles='OpenStreetMap')
    
    # Añadir marcadores para cada ubicación
    for node_id in nodos:
        lat, lon = coordenadas[node_id]
        nombre = nombres_lugares[node_id]
        tipo = nodos_info[nodos_info['node_id'] == node_id]['tipo'].values[0]
        
        color = 'red' if tipo == 'DEPOSITO' else 'blue'
        icon = 'home' if tipo == 'DEPOSITO' else 'bicycle'
        
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>{nombre}</b><br>Node ID: {node_id}",
            tooltip=nombre,
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(mapa)
    
    # Dibujar la ruta
    coordenadas_ruta = [coordenadas[node_id] for node_id in ruta]
    
    folium.PolyLine(
        locations=coordenadas_ruta,
        color='darkblue',
        weight=3,
        opacity=0.7,
        popup='Ruta del TSP'
    ).add_to(mapa)
    
    # Añadir animación de la ruta
    plugins.AntPath(
        locations=coordenadas_ruta,
        color='darkblue',
        weight=3,
        opacity=0.6,
        delay=1000
    ).add_to(mapa)
    
    # Añadir números de orden
    for orden, node_id in enumerate(ruta[:-1]):
        lat, lon = coordenadas[node_id]
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html=f'''
                <div style="
                    font-size: 12px; 
                    font-weight: bold; 
                    color: white; 
                    background-color: green; 
                    border-radius: 50%; 
                    width: 24px; 
                    height: 24px; 
                    text-align: center; 
                    line-height: 24px;
                    border: 2px solid white;
                ">
                    {orden}
                </div>
            ''')
        ).add_to(mapa)
    
    # Guardar el mapa
    archivo_mapa = 'mapa_ruta_bicimad.html'
    mapa.save(archivo_mapa)
    print(f'Mapa guardado como: {archivo_mapa}')
    
    # Abrir el mapa en el navegador
    archivo_completo = os.path.abspath(archivo_mapa)
    webbrowser.open('file://' + archivo_completo)
    print('Abriendo mapa en el navegador...')
    
else:
    print('\nNo se encontraron arcos (no hay solución válida)')

