import numpy as np
import networkx as nx
import networkx.algorithms.approximation.treewidth as tw
import matplotlib.pyplot as plt
import random



def make_graph(n,p): #Makes a random graph from the number of vertices and probablity of an edge between each vertex
    adjm = np.zeros([n,n],dtype=bool) #Adjacency matrix starts as all 0's
    for i in range(np.ma.size(adjm,0)): #For each possible edge randomly sets it to 1 with probablity p
        for j in np.array(range(np.ma.size(adjm,0)-i))+i:
            if random.random() < p and i !=j:
                adjm[i,j] = True
                adjm[j,i] = True
    return adjm


def edge_node_list(adj_matrix): #Returns a list of vertices and edges from a given adjacency matrix
    V = np.array(range(np.ma.size(adj_matrix,0)))
    E = []
    for i in range(np.ma.size(adj_matrix,0)):
        for j in np.array(range(np.ma.size(adj_matrix,0)-i))+i:
            if adj_matrix[i,j] == 1:
                E.append([i,j])
    E = np.array(E)
    return [V,E]

def makerandq(n,d,conec): #Makes an adjacency matrix representing a random quantum circuit with a given depth, number of qubits, and qubit connectivity
    d = int(d)
    qugraph = np.zeros([n,d], dtype = bool)
    eye = np.identity(n,dtype = bool)
    
    for i in range(d):
        ind = conec[random.randint(0,len(conec)-1)]
        ind1 = ind[0]
        ind2 = ind[1]
        
        qugraph[ind1, i] = True
        qugraph[ind2, i] = True
    qugraph = np.concatenate((eye,qugraph,eye), axis = 1)
    
    adj_matrix = np.zeros([d+2*n,d+2*n], dtype = bool)
    
    for i in range(d+2*n):
        for j in range(n):
            if qugraph[j,i]:
                working = True
                k = i
                while working:
                    if k == 0:
                        working = False
                    else:
                        k = k-1
                        if qugraph[j,k]:
                            adj_matrix[k,i] = True
                            adj_matrix[i,k] = True
    return adj_matrix

def plot_quantum(n,res): #Plots the how the tree width scales with increasing number of qubits as the depth scales given a max number of qubits and the resolution of the graph
    for pvalue in range(res+1): #Loops over each value for the depth of the circuit
        tws = [] #Empty list for the treewidth values to go
        for nvalue in range(3,n+1): #Loops for each number of qubits in the circuit
            conn = [] #Creats the connectivity of the quantum circuit
            for i in range(nvalue-1):
                conn.append([i,i+1]) #Linear array of qubits nearest neigbour
            temp = [] #Tempary values to be averaged over
            for trials in range(100):
                depth = nvalue**(2*pvalue/res) #Depth of the circuit ranges from 1 to n^2
                G = nx.Graph() #Creates an empty graph
                edges_nodes = edge_node_list(makerandq(nvalue,depth,conn)) #Makes a random quantum circuit with nvalue qubits and a depth of depth with conectivity conn
                G.add_nodes_from(edges_nodes[0])
                G.add_edges_from(edges_nodes[1])
                treewidth = tw.treewidth_min_degree(G)[0] #Finds the treewidth of the quantum circuit
                temp.append(treewidth)
                    
            tws.append(np.average(temp)) #Averages the treewidth
        print('n^'+str(2*pvalue/res)) #Prints the scaling of the depth as a functiong of number of qubits
        plt.plot(range(3,n+1),tws) #Plots the treewidth as a function of number of qubits
        plt.xlabel("Number of qubits")
        plt.ylabel("Treewidth")
        plt.show()

    


















