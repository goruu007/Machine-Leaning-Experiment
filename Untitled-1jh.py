print("Gaurav Raikwar")
print("Enrollment No. 0901AI223D04")
import matplotlib.pyplot as plt
import networkx as nx
# Define the graph edges
edges = [(5,3), (5,7), (3,2), (3,4), (7,8), (4,8)]
# Create a graph
G = nx.Graph()
G.add_edges_from(edges)
# Function for DFS traversal
def dfs(graph, start_node, visited=None):
    if visited is None:
        visited = set()
    visited.add(start_node)
    print(start_node, end=' ')
    for neighbor in graph.neighbors(start_node):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
# Perform DFS traversal from node 5
start_node = 5
print("DFS Traversal:")
dfs(G, start_node)
# Visualize the graph
plt.figure(figsize=(6, 4))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', font_size=12, 
font_weight='bold')
plt.title("Final DFS Traversal")
plt.show()
