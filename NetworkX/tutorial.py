import networkx as nx
G = nx.Graph()

# a graph is a collection of nodes
# identified pairs of nodes called edges, links, etc.
# nodes can be any hashable object
    # ex. text string, image, XML object, another graph, custom node object
    # None object should not be used as a node

''' ADDING NODES'''
# add one node at a time
G.add_node(1)

# add nodes from iterable container (ex. list)
G.add_nodes_from([2, 3])

# add nodes with node attributes in format (node, node_attribute_dict)
G.add_nodes_from([
    (4, {'colour': 'red'}),
    (5, {'colour': 'green'})
])

# nodes from one graph can be incorporated into another
# G now contains nodes of H as nodes of G
H = nx.path_graph(10)
G.add_nodes_from(H)

# alternatively, add H as a node in G
# now H is a graph stored within a node of G
G.add_node(H)

''' ADDING EDGES '''
# add one edge at a time
G.add_edge(1, 2)
e = (2, 3)
G.add_edge(*e)  # unpack edge tuple e

# add a list of edges
G.add_edges_from([(1, 2), (1, 3)])

# ebunch: any iterable container of edge-tuples
# an edge-tuple can be a 2-tuple of nodes, or 3-tuple with 2 nodes & attribute dict
    # ex. (2, 3, {'weight': 3.1415})
G.add_edges_from(H.edges)

# remove all nodes and edges
G.clear()

# automatically ignores duplicate nodes and edges
G.add_edges_from([(1, 2), (1, 3)])
G.add_node(1)
G.add_edge(1, 2)
G.add_node("spam")          # adds "spam"
G.add_nodes_from("spam")    # adds 4 nodes 's', 'p', 'a', 'm'
G.add_edge(3, 'm')

# now consists of 8 nodes and 3 edges
#print("number of nodes: " + str(G.number_of_nodes()))   # 8
#print("number of edges: " + str(G.number_of_edges()))   # 3

# edge can be associated with any object x
    # G.add_edge(n1, n2, object = x)
    # ex. n1, n2 are protein objects, and x is a record of their interactions
# convert_node_labels_to_integers() --> gives more traditional graph w/ int labels

''' EXAMINE ELEMENTS OF A GRAPH '''
# presents a continually updated read-only view of the graph structure
# dict-like: look up attributes via views and iterate with .items() and .data('span')
#print(list(G.nodes))    # [1, 2, 3, "spam", 's', 'p', 'a', 'm']
#print(list(G.edges))    # [(1, 2), (1, 3), (3, 'm')]
#print(list(G.adj[1]))   # [2, 3]; alt. use list(G.neighbors(1))
#print(G.degree[1])      # 2; number of edges incident to 1

# report edges and degree from a subset of all nodes using nbunch
# an nbunch is any of None (all nodes), a node, or an iterable container of nodes that is not itself a node in the graph
#print(G.edges([2, 'm']))    # [(2, 1), ('m', 3)]
#print(G.degree([2, 3]))     # [(2, 1), (3, 2)]

''' REMOVE ELEMENTS '''
# Graph.remove_node()
# Graph.remove_nodes_from()
# Graph.remove_edge()
# Graph.remove_edges_from()

''' USING GRAPH CONSTRUCTORS '''
G.clear()
G.add_edge(1, 2)
H = nx.DiGraph(G)       # create directed graph using connections from G
#print(list(H.edges()))  # [(1, 2), (2, 1)]
edgeList = [(0, 1), (1, 2), (2, 3)]
H = nx.Graph(edgeList)
#print(list(H.edges()))  # [(0, 1), (1, 2), (2, 3)]

''' ACCESSING EDGES AND NEIGHBOURS '''
# access edges and neighbours using subscript notation
G = nx.Graph([(1, 2, {'colour': 'yellow'})])
#print(G[1])             # {2: {'colour': 'yellow'}}
#print(G[1][2])          # {'colour': 'yellow'}
#print(G.edges[1, 2])    # {'colour': 'yellow'}

# get/set attributes of edge using subscript notation if it exists
G.add_edge(1, 3)
G[1][3]['colour'] = "blue"
G.edges[1, 2]['colour'] = "red"
#print(G.edges[1, 2])    # {'colour': 'red'}

# examine all (node, adjacency) pairs using G.adjacency() or G.adj.items()
# note: for undirected graphs, adjacency iteration sees each edge twice
FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
#print(FG.adj.items())   # {1: {2: {'weight': 0.125}, 3: {'weight': 0.75}}, 2: {1: {'weight': 0.125}, 4: {'weight': 1.2}}, 3: {1: {'weight': 0.75}, 4: {'weight': 0.375}}, 4: {2: {'weight': 1.2}, 3: {'weight': 0.375}}}
for n, neighbours in FG.adj.items():
    for neighbour, eattr in neighbours.items():
        weight = eattr['weight']
        if weight < 0.5: break
            #print(f"({n}, {neighbour}, {weight:.3})")
            # (1, 2, 0.125)
            # (2, 1, 0.125)
            # (3, 4, 0.375)
            # (4, 3, 0.375)

for (u, v, weight) in FG.edges.data('weight'):
    if weight < 0.5: break
        #print(f"({u}, {v}, {weight:.3})")
        # (1, 2, 0.125)
        # (3, 4, 0.375)

''' ADDING ATTRIBUTES TO GRAPHS, NODES, AND EDGES '''
# weights, labels, colours, or other Python objects can be added to graphs, nodes, or edges
# each graph, node, and edge can hold key-value attribute pairs in dict

# add graph attributes when creating a new graph
G = nx.Graph(day = "Friday")
#print(G.graph)      # {'day': 'Friday'}

# modify attributes later
G.graph['day'] = "Monday"
#print(G.graph)      # {'day': 'Monday'}

# add node attributes
G.add_node(1, time = '5pm')
G.add_nodes_from([3], time = '2pm')
#print(G.nodes[1])       # {'time': '5pm'}
G.nodes[1]['room'] = 714
#print(G.nodes.data())   # [(1, {'time': '5pm', 'room': 714}), (3, {'time': '2pm'})]

# note: adding a node to G.nodes doesn't add it to the graph
    # must use G.add_node()
    # same for edges

# special attribute weight should be numberic
G.add_edge(1, 2, weight = 4.7)
G.add_edges_from([(3, 4), (4, 5)], colour = "red")
G.add_edges_from([(1, 2, {'colour': 'blue'}), (2, 3, {'weight': 8})])
G[1][2]['weight'] = 4.7
G.edges[3, 4]['weight'] = 4.2

''' DIRECTED GRAPHS '''
# directed version of neighbors() = successors()
# degree reports sum of both in_degree and out_degree
DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
#print(DG.out_degree(1, weight = 'weight'))  # 0.5
#print(DG.degree(1, weight = 'weight'))      # 1.25, not 2
#print(list(DG.successors(1)))   # [2]
#print(list(DG.neighbors(1)))    # [2]

# don't just use directed and undirectly interchangeably
# first convert with either of:
DG.to_undirected()
H = nx.Graph(G)

''' MULTIGRAPHS '''
# multiple edges between any pair of nodes
# MultiGraph and MultiDiGraph can add same edge twice w/ diff. edge data
MG = nx.MultiGraph()
MG.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 0.75), (2, 3, 0.5)])
#print(dict(MG.degree(weight = 'weight')))   # {1: 1.25, 2: 1.75, 3: 0.5}

# however, incompatible with many algorithms
    # should convert to standard graph so measurements are well defined
    # ex. convert by taking minimum weight of duplicate edge:
GG = nx.Graph()
for n, neighbours in MG.adjacency():
    for neighbour, edict in neighbours.items():
        minvalue = min([d['weight'] for d in edict.values()])
        GG.add_edge(n, neighbour, weight = minvalue)
#print(nx.shortest_path(GG, 1, 3))   # [1, 2, 3]

''' GRAPH GENERATORS AND OPERATIONS '''
# 1. classic graph operations
# subgraph(G, nbunch) --> returns subgraph induced on nodes in nbunch
# union(G, H[, rename, name]) --> returns union of graphs G and H
# disjoint_union(G, H) --> returns disjoint union of graphs G and H
# cartesian_product(G, H) --> Returns the Cartesian product of G and H
    # every combination of pairs (item in G, item in H)
# compose(G, H) --> Returns a new graph of G composed with H
# complement(G) --> Returns the graph complement of G
    # edge becomes non-edge, non-edge becomes edge
# create_empty_copy(G[, with_data]) --> Returns a copy of the graph G with all of the edges removed
# to_undirected(graph) --> Returns an undirected view of the graph graph
# to_directed(graph) --> Returns a directed view of the graph graph

# 2. get a classic small graph
# petersen_graph([create_using]) --> Returns the Petersen graph
# tutte_graph([create_using]) --> Returns the Tutte graph
# sedgewick_maze_graph([create_using]) --> Return a small maze with a cycle
# tetrahedral_graph([create_using]) --> Return the 3-regular Platonic Tetrahedral graph

# 3. use a constructive generator for a classic graph
# complete_graph(n[, create_using]) --> Return the complete graph K_n with n nodes
# complete_bipartite_graph(n1, n2[, create_using]) --> Returns the complete bipartite graph K_{n_1,n_2}
# barbell_graph(m1, m2[, create_using]) --> Returns the Barbell Graph: two complete graphs connected by a path
# lollipop_graph(m, n[, create_using]) --> Returns the Lollipop Graph; K_m connected to P_n

# 4. use stochastic graph generator
# erdos_renyi_graph(n, p[, seed, directed]) --> Returns a Gn,p random graph, also known as an Erdős-Rényi graph or a binomial graph
# watts_strogatz_graph(n, k, p[, seed]) --> Returns a Watts–Strogatz small-world graph
# barabasi_albert_graph(n, m[, seed]) --> Returns a random graph according to the Barabási–Albert preferential attachment model
# random_lobster(n, p1, p2[, seed]) --> Returns a random lobster graph

# 5. read graph stored in a file using common graph formats
# ex. edge lists, adjacency lists, GML, GraphML, pickle, LEDA, etc.
    # nx.write_gml(red, "path.to.file")
    # mygraph = nx.read_gml("path.to.file")

''' ANALYZING GRAPHS '''
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3)])
G.add_node("spam")
# print(list(nx.connected_components(G)))         # [{1, 2, 3}, {'spam'}]
# print(sorted(deg for node, deg in G.degree()))  # [0, 1, 1, 2]
# print(nx.clustering(G))     # {1: 0, 2: 0, 3: 0, 'spam': 0}

# iterate over (node, value) 2-tuples
sp = dict(nx.all_pairs_shortest_path(G))
# print(sp[3])    # {3: [3], 1: [3, 1], 2: [3, 1, 2]}

''' DRAWING GRAPHS '''
import matplotlib.pyplot as plt
G = nx.petersen_graph()
plt.subplot(121)
nx.draw(G, with_labels = True, font_weight = 'bold')
plt.subplot(122)
nx.draw_shell(G, nlist = [range(5, 10), range(5)], with_labels = True, font_weight = 'bold')
plt.show()
# plt.savefig("graph.png")