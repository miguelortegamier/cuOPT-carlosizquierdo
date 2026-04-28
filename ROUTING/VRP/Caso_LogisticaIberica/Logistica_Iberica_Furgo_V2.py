import cudf
import cuopt
import numpy as np
import pandas as pd
import time
import cudf
from cuopt import routing

#Definir el problema y cargar datos:
flota=pd.read_csv("vehicles_fleet_130.csv")
nodos=pd.read_csv("nodes_city_2500_no_access.csv")
locations=list(range(len(nodos)))
vehicles=cudf.Series(range(len(flota)), dtype='int32')  
vehicle_types = cudf.Series(range(len(flota)), dtype='int32')
capacities=flota.loc[:,"cap_packages"]
capacities=cudf.Series(capacities.astype('int32'))
drop_return=[False]*len(vehicles)
cost_matrix=pd.read_csv("dist_km_city_2500.csv", index_col=0)
cost_matrix.columns = range(len(cost_matrix.columns)) # Resetear índices para evitar warnings de deprecación
cost_matrix.index = range(len(cost_matrix.index)) # Resetear índices para evitar warnings de deprecación
cost_matrix=cudf.DataFrame(cost_matrix)
time_matrix=pd.read_csv("time_min_city_2500.csv", index_col=0)
time_matrix.columns = range(len(time_matrix.columns)) # Resetear índices para evitar warnings de deprecación
time_matrix.index = range(len(time_matrix.index)) # Resetear índices para evitar warnings de deprecación
time_matrix=cudf.DataFrame(time_matrix, dtype='float32')
demand_vector=nodos.loc[:,"demand_packages"]
demand_vector=cudf.Series(demand_vector.astype('int32'))
fixed_costs=flota.loc[:,"cost_fix"]
fixed_costs=cudf.Series(fixed_costs.astype('float32'))
service_times=nodos.loc[:,"service_min"]
service_times=cudf.Series(service_times.astype('int32'))
earliest_time=nodos.loc[:,"ready_min"]
earliest_time=cudf.Series(earliest_time.astype('int32'))
latest_time=nodos.loc[:,"due_min"]
latest_time=cudf.Series(latest_time.astype('int32'))
partidas_vehiculos=flota.loc[:,"start_location"]
partidas_vehiculos=cudf.Series(partidas_vehiculos.astype('int32'))
llegadas_vehiculos=flota.loc[:,"return_location"]
llegadas_vehiculos=cudf.Series(llegadas_vehiculos.astype('int32'))

data_model=routing.DataModel(len(locations), len(vehicles))
data_model.add_transit_time_matrix(time_matrix)
data_model.set_vehicle_types(vehicle_types) 
data_model.add_capacity_dimension("demand", demand_vector, capacities)
data_model.set_vehicle_fixed_costs(fixed_costs)
data_model.set_drop_return_trips(cudf.Series(drop_return))
data_model.set_order_service_times(service_times)
data_model.set_order_time_windows(earliest_time, latest_time)
data_model.set_vehicle_locations(partidas_vehiculos, llegadas_vehiculos)
data_model.add_break_dimension(break_earliest=cudf.Series([150]*len(vehicles), dtype='int32'),break_latest=cudf.Series([270]*len(vehicles), dtype='int32'),break_duration=cudf.Series([60]*len(vehicles), dtype='int32'))
data_model.set_vehicle_max_times(cudf.Series([540.0]*len(vehicles), dtype='float32')) 

#Matriz de costes por vehículo:
for i, veh in flota.iterrows():
    cost_km_vehiculo = flota.loc[i, "cost_km"]
    cost_matrix_vehiculo = (cost_matrix * cost_km_vehiculo)
    cost_matrix_vehiculo = cudf.DataFrame(cost_matrix_vehiculo, dtype='float32')
    data_model.add_cost_matrix(cost_matrix_vehiculo, vehicle_type=i)

# Restricción ZBE:
for i in range(len(flota)):
    electrico = flota.loc[i, "is_electric"] == 1
    if electrico:
        orders_permitidos = cudf.Series(range(len(nodos)), dtype='int32')
    else:
        nodos_no_zbe = nodos[nodos["zbe_required"] == 0].index.tolist()
        orders_permitidos = cudf.Series(nodos_no_zbe, dtype='int32')
    data_model.add_vehicle_order_match(i, orders_permitidos)

# Resolver el problema:
inicio=time.time()
cuopt_solution=routing.Solve(data_model)
tiempo_resolucion=time.time()-inicio
cuopt_solution.display_routes()

print(f"\nMensaje del solver: {cuopt_solution.message}")
print(f"Estado: {cuopt_solution.status}")
print(f"Costo total: {cuopt_solution.total_objective_value:.4f}")
print(f"Tiempo de resolución: {tiempo_resolucion:.4f} segundos")
