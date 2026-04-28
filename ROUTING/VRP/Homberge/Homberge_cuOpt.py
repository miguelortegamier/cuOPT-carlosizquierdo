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

datos=pd.read_csv("Homberge_csv/C1_10_6.csv", sep=',')
locations=list(range(len(datos)))
capacity=1000
n_vehicles=250
vehicles=list(range(n_vehicles))
vehicles=cudf.Series(vehicles, dtype='int32')
capacities=cudf.Series([capacity]*len(vehicles), dtype='int32')
demand_vector=datos.loc[:,"DEMAND"]
demand_vector=cudf.Series(demand_vector.astype('int32'))
earliest_time=datos.loc[:,"READY TIME"]
earliest_time=cudf.Series(earliest_time.astype('int32'))
latest_time=datos.loc[:,"DUE DATE"]
latest_time=cudf.Series(latest_time.astype('int32'))
service_times=datos.loc[:,"SERVICE TIME"]
service_times=cudf.Series(service_times.astype('int32'))
cost_matrix=np.zeros((len(locations), len(locations)))
for i in locations:
    for j in locations:
        punto_a=(datos.loc[i,"XCOORD."], datos.loc[i,"YCOORD."])
        punto_b=(datos.loc[j,"XCOORD."], datos.loc[j,"YCOORD."])
        distancia=calcular_distancia(punto_a, punto_b)
        cost_matrix[i][j]=distancia

cost_matrix=cudf.DataFrame(cost_matrix)
data_model=routing.DataModel(len(locations),len(vehicles))
data_model.add_cost_matrix(cost_matrix)
data_model.add_capacity_dimension("demand", demand_vector, capacities)
data_model.set_order_time_windows(earliest_time, latest_time)
data_model.set_order_service_times(service_times)

# Multiobjetivo: distancia + penalización por usar vehículos
objectives = cudf.Series([
    routing.Objective.COST,               
    routing.Objective.VEHICLE_FIXED_COST  
])
objective_weights = cudf.Series([1.0, 1000.0], dtype="float32")

data_model.set_objective_function(objectives, objective_weights)

inicio=time.time()
settings = routing.SolverSettings()
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
