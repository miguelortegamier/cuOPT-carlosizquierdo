import cuopt
import numpy as np
import pandas as pd
import cudf
import time
from cuopt import routing
import math

def calcular_distancia(a,b):
    dist=math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    return dist

datos=pd.read_csv("XL_csv/XL-n7854  -k365.csv", sep=';')
locations=list(range(len(datos)))
capacity=datos.loc[0,"Capacidad"]
n_vehicles = len(datos)  
vehicles=list(range(n_vehicles))
vehicles=cudf.Series(vehicles, dtype='int32')
capacities=cudf.Series([capacity]*len(vehicles), dtype='int32')
demand_vector=datos.loc[:,"Demanda"]
demand_vector=cudf.Series(demand_vector.astype('int32'))
cost_matrix=np.zeros((len(locations), len(locations)))
for i in locations:
    for j in locations:
        punto_a=(datos.loc[i,"X"], datos.loc[i,"Y"])
        punto_b=(datos.loc[j,"X"], datos.loc[j,"Y"])
        distancia=calcular_distancia(punto_a, punto_b)
        cost_matrix[i][j]=distancia

cost_matrix=cudf.DataFrame(cost_matrix)
data_model=routing.DataModel(len(locations),len(vehicles))
data_model.add_cost_matrix(cost_matrix)
data_model.add_capacity_dimension("demand", demand_vector, capacities)

inicio=time.time()
settings = routing.SolverSettings()
settings.set_time_limit(7200)  
settings.set_verbose_mode(True)
cuopt_solution = routing.Solve(data_model, settings)
tiempo_resolucion=time.time()-inicio
cuopt_solution.display_routes()

print(f"\n{'='*70}")
print("RESULTADOS")
print("="*70)
print(f"Mensaje del solver: {cuopt_solution.message}")
print(f"Estado: {cuopt_solution.status}")
print(f"Costo obtenido: {cuopt_solution.total_objective_value:.4f}")
print(f"Tiempo de resolución: {tiempo_resolucion:.2f} segundos")
