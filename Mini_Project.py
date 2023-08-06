import pandas as pd
import networkx as nx
import osmnx as os
import numpy as np
from haversine import haversine
from haversine import haversine_vector
import csv

df = pd.read_csv ( 'ETM.csv' )
stops_df = pd.read_csv ( 'stops.txt' )
bus_df = pd.read_csv ( 'bus_stop.csv' )
gps_df = pd.read_csv ( 'GPS.csv', nrows = 100 )
# df1=pd.read_csv('/content/route_point.csv')
new_df = df.groupby ( ['from_bus_stop_id', 'till_bus_stop_id'] ).sum ().reset_index ()
new_df = new_df.sort_values ( by = ['total_ticket_amount'], ascending = False )
k = 0
for index, x in new_df.iterrows () :
    if x[0] == 0 or x[1] == 0 :
        new_df = new_df.drop ( index )
        k = k + 1
filter_df = pd.DataFrame ( [new_df.from_bus_stop_id, new_df.till_bus_stop_id, new_df.total_ticket_amount] ).transpose ()
top100_df = filter_df.nlargest ( 100, 'total_ticket_amount' )

total_revenue = top100_df.sum ()[2]
print ( "Ttoal revenue Earned is {}".format ( total_revenue ) )

actual_revenue = total_revenue / 0.4

print ( "Actual Revenue earned is {}".format ( actual_revenue ) )

print ( "No of Data points with 0 values in stop_id is {}".format ( k ) )

##Counting missing data points

k = 0
for index, x in df.iterrows () :
    if x[6] == 0 or x[9] == 0 or x[10] == 0 or x[1] == 0 or x[3] == 0 :
        k = k + 1

print ( "No of data points with 0 values are {}".format ( k ) )



# G = os.graph.graph_from_place ( 'Bengaluru', network_type = 'drive' )

G = os.graph.graph_from_bbox ( 13.2272, 12.7509, 78.1492, 77.3348, network_type = 'drive' )
#
for index, x in top100_df.iterrows () :
    m = float ( bus_df[bus_df.bus_stop_id == x[0]].latitude_current )
    n = float ( bus_df[bus_df.bus_stop_id == x[0]].longitude_current )
    one = os.distance.nearest_nodes ( G, n, m )

    p = float ( bus_df[bus_df.bus_stop_id == x[1]].latitude_current )
    q = float ( bus_df[bus_df.bus_stop_id == x[1]].longitude_current )
    two = os.distance.nearest_nodes ( G, q, p )

    top100_df.loc[index, 'from_lat'] = m
    top100_df.loc[index, 'from_lon'] = n

    top100_df.loc[index, 'till_lat'] = p
    top100_df.loc[index, 'till_lon'] = q

    # node1_lat=G.node[one]['lat']
    # node1_lon=G.node[one]['lon']
    lat1, lon1 = G.nodes[one]['y'], G.nodes[one]['x']

    #   node2_lat=G.node[two]['lat']
    #   node2_lon=G.node[two]['lon']
    lat2, lon2 = G.nodes[two]['y'], G.nodes[two]['x']

    try :
        dist = nx.dijkstra_path_length ( G, one, two, weight = 'length' )
        # print ( 'Distance between {} and {} is {}'.format ( one, two, dist ) )
    except :
        # print("No route Access")
        dist = None
    top100_df.loc[index, 'Distance_meters'] = dist
    try :
        top100_df.loc[index, 'Cost_rupees'] = (dist / 4500) * 95
    except :
        top100_df.loc[index, 'Cost_rupees'] = None
    try :
        top100_df.loc[index, 'Difference'] = top100_df.loc[index, 'total_ticket_amount'] - (dist / 4500) * 95
    except :
        top100_df.loc[index, 'Difference'] = None

    try :
        top100_df.loc[index, 'error_from_stop'] = haversine ( (lat1, lon1), (m, n) ) * 1000
        top100_df.loc[index, 'error_till_stop'] = haversine ( (lat2, lon2), (p, q) ) * 1000

    except :
        top100_df.loc[index, 'error_from_stop'] = None
        top100_df.loc[index, 'error_till_stop'] = None
top100_df.to_csv ( 'top100_df.csv' )





## Plots


# Maximum Revenue route

mtfi = int ( top100_df.iloc[0].from_bus_stop_id )
mtti = int ( top100_df.iloc[0].till_bus_stop_id )
mtfi_lon = float ( bus_df[bus_df.bus_stop_id == mtfi].longitude_current )
mtfi_lat = float ( bus_df[bus_df.bus_stop_id == mtfi].latitude_current )
mtti_lon = float ( bus_df[bus_df.bus_stop_id == mtti].longitude_current )
mtti_lat = float ( bus_df[bus_df.bus_stop_id == mtti].latitude_current )

os.config ( log_console = True, use_cache = True )

orig_node = os.nearest_nodes ( G, mtfi_lon, mtfi_lat )

dest_node = os.nearest_nodes ( G, mtti_lon, mtti_lat )

route_ticket = nx.shortest_path ( G, orig_node, dest_node, weight = 'length' )

route_map = os.plot_route_folium ( G, route_ticket )

route_map.save ( 'route_ticket.html' )

# Maximum cost route

max_cost = top100_df['Cost_rupees'].max ()
mcfi = int ( top100_df.loc[top100_df['Cost_rupees'] == max_cost].from_bus_stop_id )
mcti = int ( top100_df.loc[top100_df['Cost_rupees'] == max_cost].till_bus_stop_id )
mcfi_lon = float ( bus_df[bus_df.bus_stop_id == mcfi].longitude_current )
mcfi_lat = float ( bus_df[bus_df.bus_stop_id == mcfi].latitude_current )
mcti_lon = float ( bus_df[bus_df.bus_stop_id == mcti].longitude_current )
mcti_lat = float ( bus_df[bus_df.bus_stop_id == mcti].latitude_current )

os.config ( log_console = True, use_cache = True )

orig_node = os.nearest_nodes ( G, mcfi_lon, mcfi_lat )

dest_node = os.nearest_nodes ( G, mcti_lon, mcti_lat )

route_cost = nx.shortest_path ( G, orig_node, dest_node, weight = 'length' )

route_map = os.plot_route_folium ( G, route_cost )

route_map.save ( 'route_cost.html' )

# Least Revenue and  Cost difference Route

min_differ = top100_df['Difference'].min ()
midfi = int ( top100_df.loc[top100_df['Difference'] == min_differ].from_bus_stop_id )
midti = int ( top100_df.loc[top100_df['Difference'] == min_differ].till_bus_stop_id )
midfi_lon = float ( bus_df[bus_df.bus_stop_id == midfi].longitude_current )
midfi_lat = float ( bus_df[bus_df.bus_stop_id == midfi].latitude_current )
midti_lon = float ( bus_df[bus_df.bus_stop_id == midti].longitude_current )
midti_lat = float ( bus_df[bus_df.bus_stop_id == midti].latitude_current )

os.config ( log_console = True, use_cache = True )

orig_node = os.nearest_nodes ( G, midfi_lon, midfi_lat )

dest_node = os.nearest_nodes ( G, midti_lon, midti_lat )

route_differ = nx.shortest_path ( G, orig_node, dest_node, weight = 'length' )

route_map = os.plot_route_folium ( G, route_differ )

route_map.save ( 'route_differ.html' )


## Question 4

coords = list(zip(top100_df['from_lat'], top100_df['from_lon']))
coords_from = [tuple(x) for x in coords]

coords= list(zip(top100_df['till_lat'], top100_df['till_lon']))
coords_till=[tuple(x) for x in coords]

coords= list(zip(gps_df['LAT'], gps_df['LONGITUDE']))
coords_gps=[tuple(x) for x in coords]


data=haversine_vector(coords_from, coords_gps, comb=True)
data2=haversine_vector(coords_till, coords_gps,comb=True)

gps_df = pd.read_csv ( 'GPS.csv', nrows = 1000000 )

coords = list ( zip ( top100_df['from_lat'], top100_df['from_lon'] ) )
coords_from = [tuple ( x ) for x in coords]

coords = list ( zip ( top100_df['till_lat'], top100_df['till_lon'] ) )
coords_till = [tuple ( x ) for x in coords]

coords = list ( zip ( gps_df['LAT'], gps_df['LONGITUDE'] ) )
coords_gps = [tuple ( x ) for x in coords]

k = 0
try :
    data = haversine_vector ( coords_from, coords_gps, comb = True )
    data2 = haversine_vector ( coords_till, coords_gps, comb = True )
except :
    k = k + 1

# Output file path
output_file = 'Data.csv'

# Open the output file in write mode
with open ( output_file, 'w', newline = '' ) as file :
    # Create a CSV writer object
    writer = csv.writer ( file )

    # Write each row of the 2D array to the CSV file
    for row in data :
        writer.writerow ( row )

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv ( 'Data.csv', header = None )

# Check if all elements are less than 2 and create a boolean DataFrame
bool_df = df < 2

# Convert the boolean DataFrame to a binary matrix
binary_matrix = bool_df.astype ( int )

# Print the binary matrix
# print(binary_matrix)

output_file = 'Data1.csv'

# Open the output file in write mode
with open ( output_file, 'w', newline = '' ) as file :
    # Create a CSV writer object
    writer = csv.writer ( file )

    # Write each row of the 2D array to the CSV file
    for row in data2 :
        writer.writerow ( row )

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv ( 'Data1.csv', header = None )

bool_df = df < 2

# Convert the boolean DataFrame to a binary matrix
binary_matrix1 = bool_df.astype ( int )

# Print the binary matrix
# print(binary_matrix1)
result = binary_matrix + binary_matrix1

row_indices, col_indices = np.where ( result > 10000 )

count=0
# If any indices are found, print them
if len ( row_indices ) > 0 :
    for i in range ( len ( row_indices ) ) :
        print ( f"Element at row {row_indices[i]} and column {col_indices[i]} is greater than 2." )
        count=count+1


print('Total number of trips is{}'.format(count))


# Question 4, using For Loops


for index,x in top100_df.iterrows():
  count=0
  for ind,y in gps_df.iterrows():
    m = float ( bus_df[bus_df.bus_stop_id == x[0]].latitude_current )
    n = float(bus_df[bus_df.bus_stop_id == x[0]].longitude_current)
    one = os.distance.nearest_nodes ( G, n, m )

    p = float ( bus_df[bus_df.bus_stop_id == x[1]].latitude_current )
    q = float(bus_df[bus_df.bus_stop_id == x[1]].longitude_current)
    two = os.distance.nearest_nodes ( G, q, p )

    three=os.distance.nearest_nodes ( G, y[3], y[2] )
    try:
      dist1=nx.dijkstra_path_length ( G, one,three, weight = 'length' )
    except:
      dist1=6000
    try:
      dist2=nx.dijkstra_path_length ( G, two,three, weight = 'length' )
    except:
      dist2=6000

    if dist2 <=1000 or dist1<=1000:
      count=count+1

  print("Number of buses running on segments  {} and {} is {}".format(x[0], x[1], count))