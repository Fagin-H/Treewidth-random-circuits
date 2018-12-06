import numpy as np
import networkx as nx
import networkx.algorithms.approximation.treewidth as tw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objs as go
import plotly
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

def plot_quantum(n,res): #Plots the how the treewidth scales with increasing number of qubits as the depth scales given a max number of qubits and the resolution of the graph
    for pvalue in range(res+1): #Loops over each value for the depth of the circuit
        tws = [] #Empty list for the treewidth values to go
        for nvalue in range(3,n+1): #Loops for each number of qubits in the circuit
            conn = [] #Creats the connectivity of the quantum circuit
            for i in range(nvalue):
                for j in range(nvalue):
                    if i != j:
                        conn.append([i,j]) #Complete array of qubits
            temp = [] #Tempary values to be averaged over
            for trials in range(1000):
                depth = nvalue**(2*pvalue/res) #Depth of the circuit ranges from 1 to n^2
                G = nx.Graph() #Creates an empty graph
                edges_nodes = edge_node_list(makerandq(nvalue,depth,conn)) #Makes a random quantum circuit with nvalue qubits and a depth of depth with conectivity conn
                G.add_nodes_from(edges_nodes[0])
                G.add_edges_from(edges_nodes[1])
                treewidth = tw.treewidth_min_degree(G)[0] #Finds the treewidth of the quantum circuit
                temp.append(treewidth)
                    
            tws.append(np.average(temp)) #Averages the treewidth
        plt.plot(range(3,n+1),tws) #Plots the treewidth as a function of number of qubits
#        blue_patch = mpatches.Patch(color='blue', label='n^' + str(2*pvalue/res))
#        plt.legend(handles=[blue_patch])
    plt.xlabel("Number of qubits")
    plt.ylabel("Treewidth")
    plt.savefig('all.png')
    plt.show()
        
def plot_sudoquantum(n,res): #Plots how the treewidth scales with increasing number of qubits in a sudo quantum circuit represented by a random graph
    for pvalue in range(res+1): #Loops over each value for the sudo depth of the circuit
        tws = [] #Empty list for the treewidth values to go
        for nvalue in range(3,n+1): #Loops for each sudo number of qubits in the circuit
            temp = [] #Tempary values to be averaged over
            depth = int(nvalue**(2*pvalue/res)) #Sudo depth of the circuit ranges from 1 to n^2
            nov= depth+2*nvalue
            noe = nvalue+2*depth
            p = 2*noe/((nov)*(nov-1))
            for trials in range(1000):
                G = nx.Graph() #Creates an empty graph
                edges_nodes = edge_node_list(make_graph(nov,p)) #Makes a random sudo quantum circuit with nvalue qubits and a depth of depth
                G.add_nodes_from(edges_nodes[0])
                G.add_edges_from(edges_nodes[1])
                treewidth = tw.treewidth_min_degree(G)[0] #Finds the treewidth of the quantum circuit
                temp.append(treewidth)
                    
            tws.append(np.average(temp)) #Averages the treewidth
        plt.plot(range(3,n+1),tws) #Plots the treewidth as a function of number of qubits
#        blue_patch = mpatches.Patch(color='blue', label='n^' + str(2*pvalue/res))
#        plt.legend(handles=[blue_patch])
    plt.xlabel("Sudo number of qubits")
    plt.ylabel("Treewidth")
    plt.savefig('sudoall.png')
    plt.show()

    
def plot_graph(n,d): #Plots how the treewidth scales with increasing number of qubits in an IQP circuit represented by a random graph
    data = []
    for pvalue in range(d+1): #Loops over each value for the depth of the circuit
        tws = [] #Empty list for the treewidth values to go
        print(str(1+5*pvalue/d) + '*n')
        for nvalue in range(3,n+1): #Loops for each number of qubits in the circuit
            nvalue = nvalue**2
            temp = [] #Tempary values to be averaged over
            depth = int(nvalue*(1+5*pvalue/d)) #Depth of the circuit ranges from 1 to n^4
            nov= 3*nvalue
            noe = depth
            p = 2*noe/((nov)*(nov-1))
            
            print(nov,noe,p)
            for trials in range(100):
                G = nx.Graph() #Creates an empty graph
                edges_nodes = edge_node_list(make_graph(nov,p)) #Makes a random sudo quantum circuit with nvalue qubits and a depth of depth
                G.add_nodes_from(edges_nodes[0])
                G.add_edges_from(edges_nodes[1])
                treewidth = tw.treewidth_min_degree(G)[0] #Finds the treewidth of the quantum circuit
                temp.append(treewidth)
                    
            tws.append(np.average(temp)) #Averages the treewidth
            
        data.append(go.Scatter(x=[i**2 for i in range(3, n+1)], y=tws, name = 'Depth: ' + str(1+5*pvalue/d) + '*n'))
        
    layout = dict(title = 'Treewidth with diffrent scaling depths',
              xaxis = dict(title = 'Number of qubits'),
              yaxis = dict(title = 'Treewidth'),
              )

    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig) #Plots the treewidth as a function of number of qubits



def checkloop(nov,noe,nol):
    
    heatmap = np.zeros([noe,nol], dtype = float)
    
    for i in range(noe):
        for j in  range(nol):
            G = nx.Graph()
            nodes = [i for i in range(nov)]
            G.add_nodes_from(nodes)
            
            for i in range(i):
                a = random.randint(0,nov-1)
                b = random.randint(0,nov-1)
                
                while a == b:
                    b = random.randint(0,nov-1)
                    
                G.add_edge(a,b)
                
            tw1 = tw.treewidth_min_degree(G)[0]
            
            for i in range(j):
                a = random.randint(0,nov-1)
                G.add_edge(a,a)
            
            tw2 = tw.treewidth_min_degree(G)[0]
            
            if tw1 == 0:
                heatmap[i,j] = 0
            else:
                heatmap[i,j] = tw2/tw1
            
    heatmap = go.Heatmap(z = heatmap)
    data=[heatmap]
    plotly.offline.plot(data)














