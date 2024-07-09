print("Gaurav Raikwar")
print("Enrollment No. 0901AI223D04")
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
# Define the graph edges
edges = [(5,3), (5,7), (3,2), (3,4), (7,8), (4,8)]
# Create a graph
G = nx.Graph()
G.add_edges_from(edges)
# Function for BFS traversal
def bfs(graph, start_node):
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    while queue:
        current_node = queue.popleft()
        print(current_node, end=' ')
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
# Perform BFS traversal from node 5
start_node = 5
print("BFS Traversal:")
bfs(G, start_node)
# Visualize the graph
plt.figure(figsize=(6, 4))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', font_size=12, 
font_weight='bold')
plt.title("Final BFS Traversal")
plt.show()
