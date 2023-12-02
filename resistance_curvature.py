import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
#from load_file import load_citation
#import dgl
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    #adj = adj + sp.eye(adj.shape[0])
    # adj = adj @ adj
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocsr()
def construct_matrices(A):
    """ Enumerates the edges in the graph and constructs the matrices neccessary for the algorithm.

    Parameters:
    -----------
    A : sp.csr_matrix, shape [N, N]
        The graph adjacency matrix.

    Returns:
    --------
    L : sp.csr_matrix, shape [N, N]
        The graph laplacian, unnormalized.
    W_sqrt : sp.coo_matrix, shape [e, e]
        Diagonal matrix containing the square root of weights of each edge.
    B : sp.coo_matrix, shape [e, N]
        Signed vertex incidence matrix.
    edges : tuple
        A tuple of lists containing the row and column indices of the edges.
    """
    L = sp.csgraph.laplacian(A)
    rows, cols = A.nonzero()
    
    weights = np.array(A[rows, cols].tolist())
    W_sqrt = sp.diags(weights, [0])
    # Construct signed edge incidence matrix
    num_vertices = A.shape[0]
    num_edges = W_sqrt.shape[0]
    assert (num_edges == len(rows) and num_edges == len(cols))
    B = sp.coo_matrix((
        ([1] * num_edges) + ([-1] * num_edges),
        (list(range(num_edges)) * 2, list(rows) + list(cols))
    ), shape=[num_edges, num_vertices])
    return L.tocsr(), W_sqrt, B, (rows, cols)


def compute_Z(L, W_sqrt, B, epsilon=1e-1, eta=1e-3, max_iters=1000, convergence_after=10,
              tolerance=1e-2, log_every=10, compute_exact_loss=False):
    """ Computes the Z matrix using gradient descent.

    Parameters:
    -----------
    L : sp.csr_matrix, shape [N, N]
        The graph laplacian, unnormalized.
    W_sqrt : sp.coo_matrix, shape [e, e]
        Diagonal matrix containing the square root of weights of each edge.
    B : sp.coo_matrix, shape [e, N]
        Signed vertex incidence matrix.
    epsilon : float
        Tolerance for deviations w.r.t. spectral norm of the sparsifier. Smaller epsilon lead to a higher
        dimensionality of the Z matrix.
    eta : float
        Step size for the gradient descent.
    max_iters : int
        Maximum number of iterations.
    convergence_after : int
        If the loss did not decrease significantly for this amount of iterations, the gradient descent will abort.
    tolerance : float
        The minimum amount of energy decrease that is expected for iterations. If for a certain number of iterations
        no overall energy decrease is detected, the gradient descent will abort.
    log_every : int
        Log the loss after each log_every iterations.
    compute_exact_loss : bool
        Only for debugging. If set it computes the actual pseudo inverse without down-projection and checks if
        the pairwise distances in Z's columns are the same with respect to the forbenius norm.

    Returns:
    --------
    Z : ndarray, shape [k, N]
        Matrix from which to efficiently compute approximate resistances.
    """
    k = int(np.ceil(np.log(B.shape[1] / epsilon ** 2)))
    # Compute the random projection matrix
    Q = (2 * np.random.randint(2, size=(k, B.shape[0])) - 1).astype(np.float)
    Q *= 1 / np.sqrt(k)
    Y = W_sqrt.dot(B).tocsr()
    Y_red = sp.csr_matrix.dot(Q, Y)

    if compute_exact_loss:
        # Use exact effective resistances to track actual similarity of the pairwise distances
        L_inv = np.linalg.pinv(L.todense())
        Z_gnd = sp.csr_matrix.dot(Y, L_inv)
        pairwise_dist_gnd = Z_gnd.T.dot(Z_gnd)

    # Use gradient descent to solve for Z
    Z = np.random.randn(k, L.shape[1])
    best_loss = np.inf
    best_iter = np.inf
    for it in range(max_iters):
        residual = Y_red - sp.csr_matrix.dot(Z, L)
        loss = np.linalg.norm(residual)
        if it % log_every == 0:
            #print(f'Loss before iteration {it}: {loss}')
            if compute_exact_loss:
                pairwise_dist = Z.T.dot(Z)
                exact_loss = np.linalg.norm(pairwise_dist - pairwise_dist_gnd)
                print(f'Loss w.r.t. exact pairwise distances {exact_loss}')

        if loss + tolerance < best_loss:
            best_loss = loss
            best_iter = it
        elif it > best_iter + convergence_after:
            # No improvement for 10 iterations
            #print(f'Convergence after {it - 1} iterations.')
            break

        Z += eta * L.dot(residual.T).T
    return Z


def compute_effective_resistances(Z, edges,A):
    """ Computes the effective resistance for each edge in the graph.

    Paramters:
    ----------
    Z : ndarray, shape [k, N]
        Matrix from which to efficiently compute approximate effective resistances.
    edges : tuple
        A tuple of lists indicating the row and column indices of edges.

    Returns:
    --------
    R : ndarray, shape [e]
        Effective resistances for each edge.
    """
    rows, cols = edges
    assert (len(rows) == len(cols))
    R = []
    B = sp.lil_matrix(A.shape)
    # Compute pairwise distances
    for i, j in zip(rows, cols):
        R.append(np.linalg.norm(Z[:, i] - Z[:, j]) ** 2)
        B[i, j] = np.linalg.norm(Z[:, i] - Z[:, j]) ** 2
    return np.array(R),B


def sparsify(A, q, R, edges, prevent_vertex_blow_up=False):
    """ Spamples a sparsifier of the graph represented by an adjacency matrix.

    Paramters:
    ----------
    A : sp.csr_matrix
        The adjacency matrix of the graph.
    q : int
        The number of samples for the sparsifier.
    R : ndarray, shape [e]
        Effective resistances (approximate) for each edge.
    edges : tuple
        A tuple of lists indicating the row and column indices of edges.
    prevent_vertex_blow_up : bool
        If the probabilities will be tweaked in order to ensure that the vertices are not
        blown up too much. Note that this will only guarantee a spectral closeness up
        to a factor of 2 * epsilon.

    Returns:
    --------
    B : sp.csr_matrix
        The adjacency matrix of the sparsified graph with at most q edges.
    """
    drop = False
    random_drop = True
    rows, cols = edges
    weights = np.array(A[rows, cols].tolist())[0, :]
    probs = weights * R
    if drop or not random_drop:
        probs /= sum(probs)

    '''
    sorted_indices = np.argsort(probs)[::-1]
    new_size = int(len(probs)*0.9)
    sampled = sorted_indices[:new_size]
    probs_new = np.sort(probs)[:new_size]
    probs_new /= sum(probs_new)
    '''

    if drop:
        k = int(probs.shape[0]*0.9)
        N = probs.shape[0]
        sorted_indices = np.argsort(probs)
        top_k_indices = sorted_indices[-k:]
        new_prob_array = np.zeros(N)
        new_prob_array[top_k_indices] = probs[top_k_indices]
        new_prob_array = new_prob_array / sum(new_prob_array)
    if random_drop:
        k = int(probs.shape[0] * 0.8)
        N = probs.shape[0]
        indices = np.random.choice(probs.shape[0],k, replace=False)
        new_prob_array = np.zeros(N)
        new_prob_array[indices] = probs[indices]
        new_prob_array = new_prob_array / sum(new_prob_array)
        print(np.count_nonzero(new_prob_array))
    if prevent_vertex_blow_up:  # Proabilities do not sum up to one? But also in the paper I guess...
        degrees = A.sum(1)[np.array(edges)].squeeze().T
        mins = 1 / np.min(degrees, axis=1) / A.shape[0]
        probs += mins
        probs /= 2
    B = sp.lil_matrix(A.shape)

    print("---------sampled----------")
    #sampled = np.random.choice(sampled, q, p=probs_new)
    if drop or random_drop:
        sampled = np.random.choice(probs.shape[0], q, p=new_prob_array)
    else:
        sampled = np.random.choice(probs.shape[0], q, p=probs.ravel())
    print("---------sampled----------")
    for idx in sampled:
        i, j = rows[idx], cols[idx]
        if drop or random_drop:
            B[i, j] += weights[idx] / q / new_prob_array[idx]
        else:
            B[i, j] += weights[idx] / q / probs[idx]
    return B.tocsr()

def calculate_curvature(A,B,edges):
    node_curvature_use = B.dot(A)
    node_curvature = node_curvature_use.tocsr()
    node_curvature_list = node_curvature.diagonal()
    
    edge_curvature = []
    for edge in edges:
        i,j = edge
        edge_curvature.append((node_curvature_list[i]-node_curvature_list[j])/B[i,j])
    edge_curvature_vector = np.array(edge_curvature)
    return edge_curvature_vector,node_curvature_list
def calculate_curvature_direct(A,B,edges):
    '''
    A = A.T
    node_curvature_use = B@A*0.5
    node_curvature = np.eye(A.shape[0])-node_curvature_use
    node_curvature_list = node_curvature.diagonal().A
    node_curvature_arrary = node_curvature_list.squeeze()
    edges = np.array(list(edges))
    edge_curvature = []
    A = A.T
    for edge in edges:
        i,j = edge
        edge_curvature.append(2*(node_curvature_arrary[i]+node_curvature_arrary[j])/A[i,j])
    edge_curvature_vector = np.array(edge_curvature)
    '''
    h_0 = A@B
    h_use = h_0
    for i in range(1):
        h_use = A@B*0.8+0.2*h_0
    #node_curvature = np.eye(A.shape[0])-h_use*0.5
    node_curvature_list = h_use.diagonal().A
    node_curvature_arrary = node_curvature_list.squeeze()
    edges = np.array(list(edges))
    edge_curvature = []
    
    for edge in edges:
        i,j = edge
        edge_curvature.append((node_curvature_arrary[i]-node_curvature_arrary[j])/B[i,j])
    edge_curvature_vector = np.array(edge_curvature)
    
    
    return edge_curvature_vector,node_curvature_arrary 
    
def spectral_sparsify(A, q=None, epsilon=1e-1, eta=1e-3, max_iters=1000, convergence_after=100,
                      tolerance=1e-2, log_every=10, prevent_vertex_blow_up=False):
    """ Computes a spectral sparsifier of the graph given by an adjacency matrix.

    Parameters:
    ----------
    A : sp.csr_matrix, shape [N, N]
        The adjacency matrix of the graph.
    q : int or None
        The number of samples for the sparsifier. If None q will be set to N * log(N) / (epsilon * 2)
    epsilon : float
        The desired spectral similarity of the sparsifier.
    eta : float
        Step size for the gradient descent when computing resistances.
    max_iters : int
        Maximum number of iterations when computing resistances.
    convergence_after : int
        If the loss did not decrease significantly for this amount of iterations, the gradient descent will abort.
    tolerance : float
        The minimum amount of energy decrease that is expected for iterations. If for a certain number of iterations
        no overall energy decrease is detected, the gradient descent will abort.
    log_every : int
        Log the loss after each log_every iterations when computing resistances.
    prevent_vertex_blow_up : bool
        If the probabilities will be tweaked in order to ensure that the vertices are not
        blown up too much. Note that this will only guarantee a spectral closeness up
        to a factor of 2 * epsilon.

    Returns:
    --------
    H : sp.csr_matrix, shape [N, N]
        Sparsified graph with at most q edges.
    """
    if q is None:
        q = int(np.ceil(A.shape[0] * np.log(A.shape[0]) / (epsilon ** 2)))

    L, W_sqrt, B, edges = construct_matrices(A)
    Z = compute_Z(L, W_sqrt, B, epsilon=epsilon, log_every=log_every, max_iters=max_iters,
                  convergence_after=convergence_after, eta=eta, tolerance=tolerance, compute_exact_loss=False)
    R = compute_effective_resistances(Z, edges)
    return sparsify(A, q, R, edges, prevent_vertex_blow_up=prevent_vertex_blow_up)
def calculate_resistance(A, q=None, epsilon=1e-1, eta=1e-3, max_iters=1000, convergence_after=100,
                      tolerance=1e-2, log_every=10, prevent_vertex_blow_up=False):
    L, W_sqrt, B, edges = construct_matrices(A)
    Z = compute_Z(L, W_sqrt, B, epsilon=epsilon, log_every=log_every, max_iters=max_iters,
                  convergence_after=convergence_after, eta=eta, tolerance=tolerance, compute_exact_loss=False)
    R,R_matrix = compute_effective_resistances(Z, edges,A)
    return R,R_matrix
def resistance_curvature(A,R_matrix,edges):
    
    edge_curvature_vector,node_curvature = calculate_curvature(A,R_matrix,edges)
    return edge_curvature_vector,node_curvature
def resistance_curvature_direct(G,R):
    A = nx.adjacency_matrix(G).todense()
    edge_curvature_vector,node_curvature = calculate_curvature_direct(A,R,G.edges)
    return edge_curvature_vector,node_curvature
def get_edge_resistance(G,R):
    edges = np.array(list(G.edges))
    edge_resistance = []
    A = A.transpose()
    for edge in edges:
        i,j = edge
        edge_resistance.append(R[i,j])
    edge_resistance_vector = np.array(edge_resistance)
    return edge_resistance_vector
def visualize_graph(A, graph_index):
    rows, cols = A.nonzero()
    plot_labels=True
    edges = list(zip(rows.tolist(), cols.tolist()))
    gr = nx.Graph()
    gr.add_edges_from(edges)
    
    pos = nx.spring_layout(gr)
    #nx.draw_networkx_nodes(gr, pos)
    #nx.draw_networkx_labels(gr, pos)
    #nx.draw_networkx_edges(gr, pos, edge_color=[d['weight'] for _, _, d in G.edges(data=True)], width=2, edge_cmap=plt.cm.Blues)
    #sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=2))
    #sm._A = []
    #plt.colorbar(sm)
    #nx.draw_networkx_nodes(gr, graph_pos, node_size=2)
    #nx.draw_networkx_edges(gr, graph_pos)
    #weights = A[rows, cols].tolist()[0]
    nx.draw(gr,node_size=10,with_labels=plot_labels)
    #if plot_labels: nx.draw_networkx_edge_labels(gr, graph_pos, edge_labels=dict(zip(edges, weights)))
    plt.show()
    base = "/root/Batch-Ollivier-Ricci-Flow-main/img/"
    save_pig = base+str(graph_index)+".png"
    plt.savefig(save_pig)
    plt.close()
def vis(G,R,N,number):
    
    bs = []
    for edge in G.edges:
        i,j = edge
        bs.append(R[i,j])

    
    for (u,v),weight in zip(G.edges,bs):
        G[u][v]['weight'] = weight
   
    node_attributes = nx.get_node_attributes(G, 'attribute')

    node_attributes = nx.get_node_attributes(G, 'attribute')
    node_colors = list(node_attributes.values())

    # 定义边的权重
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())

    # 定义节点的布局
    pos = nx.spring_layout(G)

    # 绘制节点，并根据节点属性值设置节点颜色
    nodes = nx.draw_networkx_nodes(G, pos, node_size=50)
    edges = nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_weights, edge_cmap=plt.cm.Blues)

    # 添加节点颜色栏


    # 添加边的颜色栏
    plt.colorbar(edges, label='BS')
    graph_index = 0
    # 显示图形
    print("-------------painting------------------")
    print(number)
    base = "/root/Batch-Ollivier-Ricci-Flow-main/img/"
    save_pig = base+str(graph_index)+str(number)+".png"
    print(save_pig)
    plt.savefig(save_pig)
    plt.close()
def spectral_closeness(L_A, L_X, samples=10000):
    """ Checks the spectral closeness for random vectors.

    Parameters:
    -----------
    L_A : sp.csr_matrix
        Laplacian of the original graph.
    L_A : sp.csr_matrix
        Laplacian of the sparsifier.
    samples : int
        The number of random vectors to sample.

    Returns:
    --------
    closeness : ndarray, [samples]
        The closeness of the spectral forms w.r.t. to each sample.
    """
    results = []
    for _ in range(samples):
        x = np.random.rand(L_A.shape[0])
        energy = sp.csr_matrix.dot(x.T, L_A).dot(x)
        energy_approx = sp.csr_matrix.dot(x.T, L_X).dot(x)
        results.append(np.abs((energy_approx / energy) - 1))
    return np.array(results)

def is_symmetric(A):
    if A.shape[0] != A.shape[1]:
        return False
    for i in range(A.shape[0]):
        row_start = A.indptr[i]
        row_end = A.indptr[i+1]
        for j_idx in range(row_start, row_end):
            j = A.indices[j_idx]
            val = A.data[j_idx]
            if i == j:
                continue
            if A[j, i] != val:
                return False
    return True
def sort_and_select_edge(edge,edge_curvature_vector):
    

    edge = np.array(list(edge))
    sorted_indices = np.argsort(edge_curvature_vector)
    #sorted_edges = [edge[i] for i in sorted_indices]
    sorted_edges = []
    for i in sorted_indices:
        sorted_edges.append(edge[i])
    return sorted_edges
def sort_and_select_node(node,node_curvature_vector):
    
    
    sorted_indices = np.argsort(node_curvature_vector)
    #sorted_edges = [edge[i] for i in sorted_indices]
    
    sorted_nodes = node[sorted_indices]
    
    return sorted_nodes
def get_edge_resistance(edges,R):
    edge_resistance = []
    for edge in edges:
        i,j = edge
        edge_resistance.append(R[i,j])
    return edge_resistance
def sort_edge_by_resistance(edges,R):
    edges = np.array(list(edges))
    sorted_indices = np.argsort(np.array(R))
    sort_edges = []
    for i in sorted_indices:
        sort_edges.append(edges[i])
    return sort_edges
def calculate_ling_resistance(G):
    G = G.to_undirected()
    self_laplacian = nx.laplacian_matrix(G).todense()
    self_pinv = np.linalg.pinv(self_laplacian, hermitian=True)
    pinv_diagonal = np.diag(self_pinv)
    resistance_matrix = np.expand_dims(pinv_diagonal,0) + np.expand_dims(pinv_diagonal,1)  - 2*self_pinv
    return resistance_matrix
def total_min_resistance(R,edges):
    total_R = 0
    min_R = 999999
    for edge in edges:
        i,j = edge
        total_R+=R[i,j]
        if R[i,j]==0:
            continue
        else:
            min_R = min(min_R,R[i,j])
    return total_R,min_R
import os
import ot
import time
import torch
import pathlib
import numpy as np
import pandas as pd
import multiprocessing as mp
import networkx as nx
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)
from torch_geometric.datasets import TUDataset
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy import linalg
from scipy.sparse import diags

def _preprocess_data(data, is_undirected=False):
    # Get necessary data information
    N = data.x.shape[0]
    m = data.edge_index.shape[1]

    # Compute the adjacency matrix
    if not "edge_type" in data.keys:
        edge_type = np.zeros(m, dtype=int)
    else:
        edge_type = data.edge_type

    # Convert graph to Networkx
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()
    
    return G, N, edge_type


def borf3(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    batch_add=4,
    batch_remove=2,
    device=None,
    save_dir='rewired_graphs',
    dataset_name=None,
    graph_index=0,
    debug=False
):
    print("--------resistance----------")
    # Check if there is a preprocessed graph
    dirname = f'{save_dir}/{dataset_name}'
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    edge_index_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_index_{graph_index}.pt')
    edge_type_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_type_{graph_index}.pt')
    print(edge_index_filename)
    if(os.path.exists(edge_index_filename) and os.path.exists(edge_type_filename)):
        print("--------exisit----------")
        if(debug) : print(f'[INFO] Rewired graph for {loops} iterations, {batch_add} edge additions and {batch_remove} edge removal exists...')
        with open(edge_index_filename, 'rb') as f:
            edge_index = torch.load(f)
        with open(edge_type_filename, 'rb') as f:
            edge_type = torch.load(f)
        return edge_index, edge_type

    # Preprocess data
    G, N, edge_type = _preprocess_data(data)
    adj = nx.adjacency_matrix(G)
    
    R,R_matrix = calculate_resistance(adj, epsilon=5e-1, log_every=10, convergence_after=100, eta=1e-3, max_iters=100,
                          prevent_vertex_blow_up=False)
    for _ in range(loops):
        """
        # Compute ORC
        orc = OllivierRicci(G, alpha=0)
        orc.compute_ricci_curvature()
        _C = sorted(orc.G.edges, key=lambda x: orc.G[x[0]][x[1]]['ricciCurvature']['rc_curvature'])
        """
        adj = nx.adjacency_matrix(G)
    # Rewiring begins
        print("---------------compute resistance_curvature -----------------")
        edge_curvature_vector,node_curvature = resistance_curvature(adj,R_matrix,G.edges)
       
        resistance_matrix = R_matrix.todense()
        
    
        print("---------------finish -----------------")
        _C = sort_and_select_edge(G.edges,edge_curvature_vector)
        node_list = np.arange(N)
        _N = sort_and_select_node(node_list,node_curvature)
        # Get top negative and positive curved edges
        most_pos_edges = _C[-batch_remove:]
        most_neg_edges = _C[:batch_add]
        most_connected_nodes = _N[-batch_remove:]
        '''
        # Add edges
        for (u, v) in most_neg_edges:
            pi = orc.G[u][v]['ricciCurvature']['rc_transport_cost']
            p, q = np.unravel_index(pi.values.argmax(), pi.values.shape)
            p, q = pi.index[p], pi.columns[q]
            if(p != q and not G.has_edge(p, q)):
                G.add_edge(p, q)
        '''
        for (u, v) in most_neg_edges:
            '''
            u_neighbors = list(G.neighbors(u)) + [u]
            v_neighbors = list(G.neighbors(v)) + [v]
            if random_select:
                prob_p = softmax(node_curvature[u_neighbors])
                p = np.random.choice(
                    node_curvature[u_neighbors], p=prob_p)
                p = np.where(node_curvature==p)[0][0]
                prob_q = softmax(node_curvature[v_neighbors])
                q = np.random.choice(
                    node_curvature[v_neighbors], p=prob_q)
                q = np.where(node_curvature==q)[0][0]
                if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            else:
                p = np.max(node_curvature[u_neighbors])
                p = np.where(node_curvature==p)[0][0]
                q = np.max(node_curvature[v_neighbors])
                q = np.where(node_curvature==q)[0][0]
                if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            '''
            u_neighbors = list(G.neighbors(u))+[u]
            v_neighbors = list(G.neighbors(v))+[v]
            candidates = []
            candidates_resistance = []
            if len(u_neighbors)==0 or len(v_neighbors)==0:
                continue
            for i in u_neighbors:
                for j in v_neighbors:
                    if (i != j) and (not G.has_edge(i, j)):
                        candidates.append((i, j))
                        candidates_resistance.append(resistance_matrix[i,j])
            if len(candidates)==0:
                continue
            sorted_indices = np.argsort(np.array(candidates_resistance))[-1]
            p,q = candidates[sorted_indices]
            if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            
        # Remove edges
        if G.size()<batch_remove:
            continue
        
        for node_select in most_connected_nodes:
            
            neighbors = list(G.neighbors(node_select))
            if len(neighbors)<=1:
                continue
            #remove_edges = sort_and_select_node(node_list[neighbors],node_curvature[neighbors])[-1]
            
            remove_edges = sort_and_select_node(node_list[neighbors],resistance_matrix.A[node_select][neighbors])[0]
            if(G.has_edge(node_select, remove_edges)):
                G.remove_edge(node_select, remove_edges)

    edge_index = from_networkx(G).edge_index
    edge_type = torch.zeros(size=(len(G.edges),)).type(torch.LongTensor)
    # edge_type = torch.tensor(edge_type)

    if(debug) : print(f'[INFO] Saving edge_index to {edge_index_filename}')
    with open(edge_index_filename, 'wb') as f:
        torch.save(edge_index, f)

    if(debug) : print(f'[INFO] Saving edge_type to {edge_type_filename}')
    with open(edge_type_filename, 'wb') as f:
        torch.save(edge_type, f)

    return edge_index, edge_type
def gerc2(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    batch_add=4,
    batch_remove=2,
    device=None,
    save_dir='rewired_graphs_gerc2',
    dataset_name=None,
    graph_index=0,
    debug=False
):
    print("--------resistance----------")
    # Check if there is a preprocessed graph
    dirname = f'{save_dir}/{dataset_name}'
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    edge_index_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_index_{graph_index}.pt')
    edge_type_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_type_{graph_index}.pt')
    print(edge_index_filename)
    if(os.path.exists(edge_index_filename) and os.path.exists(edge_type_filename)):
        print("--------exisit----------")
        if(debug) : print(f'[INFO] Rewired graph for {loops} iterations, {batch_add} edge additions and {batch_remove} edge removal exists...')
        with open(edge_index_filename, 'rb') as f:
            edge_index = torch.load(f)
        with open(edge_type_filename, 'rb') as f:
            edge_type = torch.load(f)
        return edge_index, edge_type

    # Preprocess data
    G, N, edge_type = _preprocess_data(data)
    
    for _ in range(loops):
        """
        # Compute ORC
        orc = OllivierRicci(G, alpha=0)
        orc.compute_ricci_curvature()
        _C = sorted(orc.G.edges, key=lambda x: orc.G[x[0]][x[1]]['ricciCurvature']['rc_curvature'])
        """
        G = G.to_undirected()
        self_laplacian = nx.laplacian_matrix(G).todense()
        self_pinv = np.linalg.pinv(self_laplacian, hermitian=True)
        pinv_diagonal = np.diag(self_pinv)
        resistance_matrix = np.expand_dims(pinv_diagonal,0) + np.expand_dims(pinv_diagonal,1)  - 2*self_pinv
    # Rewiring begins
        
        edge_curvature_vector,node_curvature = resistance_curvature_direct(G,resistance_matrix)
        
        _C = sort_and_select_edge(G.edges,edge_curvature_vector)
        node_list = np.arange(N)
        _N = sort_and_select_node(node_list,node_curvature)
        # Get top negative and positive curved edges
        most_pos_edges = _C[-batch_remove:]
        most_neg_edges = _C[:batch_add]
        most_connected_nodes = _N[-batch_remove:]
        '''
        # Add edges
        for (u, v) in most_neg_edges:
            pi = orc.G[u][v]['ricciCurvature']['rc_transport_cost']
            p, q = np.unravel_index(pi.values.argmax(), pi.values.shape)
            p, q = pi.index[p], pi.columns[q]
            if(p != q and not G.has_edge(p, q)):
                G.add_edge(p, q)
        '''
        random_select = False
        # Add edges
        for (u, v) in most_neg_edges:
            '''
            u_neighbors = list(G.neighbors(u)) + [u]
            v_neighbors = list(G.neighbors(v)) + [v]
            if random_select:
                prob_p = softmax(node_curvature[u_neighbors])
                p = np.random.choice(
                    node_curvature[u_neighbors], p=prob_p)
                p = np.where(node_curvature==p)[0][0]
                prob_q = softmax(node_curvature[v_neighbors])
                q = np.random.choice(
                    node_curvature[v_neighbors], p=prob_q)
                q = np.where(node_curvature==q)[0][0]
                if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            else:
                p = np.max(node_curvature[u_neighbors])
                p = np.where(node_curvature==p)[0][0]
                q = np.max(node_curvature[v_neighbors])
                q = np.where(node_curvature==q)[0][0]
                if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            '''
            u_neighbors = list(G.neighbors(u))
            v_neighbors = list(G.neighbors(v))
            candidates = []
            candidates_resistance = []
            if len(u_neighbors)==0 or len(v_neighbors)==0:
                continue
            for i in u_neighbors:
                for j in v_neighbors:
                    if (i != j) and (not G.has_edge(i, j)):
                        candidates.append((i, j))
                        candidates_resistance.append(resistance_matrix[i,j])
            if len(candidates)==0:
                continue
            sorted_indices = np.argsort(np.array(candidates_resistance))[-1]
            p,q = candidates[sorted_indices]
            if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            
        # Remove edges
        if G.size()<batch_remove:
            continue
        
        for node_select in most_connected_nodes:
            
            neighbors = list(G.neighbors(node_select))
            if len(neighbors)<=1:
                continue
            remove_edges = sort_and_select_node(node_list[neighbors],node_curvature[neighbors])[-1]
            #remove_edges = sort_and_select_node(node_list[neighbors],resistance_matrix[node_select][neighbors])[0]
            if(G.has_edge(node_select, remove_edges)):
                G.remove_edge(node_select, remove_edges)
        '''
        for (u, v) in most_pos_edges:
            if(G.has_edge(u, v)):
                G.remove_edge(u, v)
        '''
    edge_index = from_networkx(G).edge_index
    edge_type = torch.zeros(size=(len(G.edges),)).type(torch.LongTensor)
    # edge_type = torch.tensor(edge_type)

    if(debug) : print(f'[INFO] Saving edge_index to {edge_index_filename}')
    with open(edge_index_filename, 'wb') as f:
        torch.save(edge_index, f)

    if(debug) : print(f'[INFO] Saving edge_type to {edge_type_filename}')
    with open(edge_type_filename, 'wb') as f:
        torch.save(edge_type, f)

    return edge_index, edge_type
def gerc3(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    batch_add=4,
    batch_remove=2,
    device=None,
    save_dir='rewired_graphs_gerc3',
    dataset_name=None,
    graph_index=0,
    debug=False
):
    print("--------resistance----------")
    # Check if there is a preprocessed graph
    dirname = f'{save_dir}/{dataset_name}'
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    edge_index_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_index_{graph_index}.pt')
    edge_type_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_type_{graph_index}.pt')
    print(edge_index_filename)
    if(os.path.exists(edge_index_filename) and os.path.exists(edge_type_filename)):
        print("--------exisit----------")
        if(debug) : print(f'[INFO] Rewired graph for {loops} iterations, {batch_add} edge additions and {batch_remove} edge removal exists...')
        with open(edge_index_filename, 'rb') as f:
            edge_index = torch.load(f)
        with open(edge_type_filename, 'rb') as f:
            edge_type = torch.load(f)
        return edge_index, edge_type
    if graph_index != 0:
        exit()
    # Preprocess data
    G, N, edge_type = _preprocess_data(data)
    visualize_graph(nx.adjacency_matrix(G),graph_index)
    for _ in range(loops):
        """
        # Compute ORC
        orc = OllivierRicci(G, alpha=0)
        orc.compute_ricci_curvature()
        _C = sorted(orc.G.edges, key=lambda x: orc.G[x[0]][x[1]]['ricciCurvature']['rc_curvature'])
        """
        G = G.to_undirected()
        self_laplacian = nx.laplacian_matrix(G).todense()
        self_pinv = np.linalg.pinv(self_laplacian, hermitian=True)
        pinv_diagonal = np.diag(self_pinv)
        resistance_matrix = np.expand_dims(pinv_diagonal,0) + np.expand_dims(pinv_diagonal,1)  - 2*self_pinv
    # Rewiring begins
        
        edge_curvature_vector,node_curvature = resistance_curvature_direct(G,resistance_matrix)
      
        vis(G,resistance_matrix,N,_)
        _C = sort_and_select_edge(G.edges,edge_curvature_vector)
        node_list = np.arange(N)
        _N = sort_and_select_node(node_list,node_curvature)
        # Get top negative and positive curved edges
        most_pos_edges = _C[-batch_remove:]
        most_neg_edges = _C[:batch_add]
        most_connected_nodes = _N[-batch_remove:]
        '''
        # Add edges
        for (u, v) in most_neg_edges:
            pi = orc.G[u][v]['ricciCurvature']['rc_transport_cost']
            p, q = np.unravel_index(pi.values.argmax(), pi.values.shape)
            p, q = pi.index[p], pi.columns[q]
            if(p != q and not G.has_edge(p, q)):
                G.add_edge(p, q)
        '''
        random_select = False
        # Add edges
        for (u, v) in most_neg_edges:
            '''
            u_neighbors = list(G.neighbors(u)) + [u]
            v_neighbors = list(G.neighbors(v)) + [v]
            if random_select:
                prob_p = softmax(node_curvature[u_neighbors])
                p = np.random.choice(
                    node_curvature[u_neighbors], p=prob_p)
                p = np.where(node_curvature==p)[0][0]
                prob_q = softmax(node_curvature[v_neighbors])
                q = np.random.choice(
                    node_curvature[v_neighbors], p=prob_q)
                q = np.where(node_curvature==q)[0][0]
                if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            else:
                p = np.max(node_curvature[u_neighbors])
                p = np.where(node_curvature==p)[0][0]
                q = np.max(node_curvature[v_neighbors])
                q = np.where(node_curvature==q)[0][0]
                if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            '''
            u_neighbors = list(G.neighbors(u))+[u]
            v_neighbors = list(G.neighbors(v))+[v]
            candidates = []
            candidates_resistance = []
            if len(u_neighbors)==0 or len(v_neighbors)==0:
                continue
            for i in u_neighbors:
                for j in v_neighbors:
                    if (i != j) and (not G.has_edge(i, j)):
                        candidates.append((i, j))
                        candidates_resistance.append(resistance_matrix[i,j])
            if len(candidates)==0:
                continue
            sorted_indices = np.argsort(np.array(candidates_resistance))[-1]
            p,q = candidates[sorted_indices]
            if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            
        # Remove edges
        if G.size()<batch_remove:
            continue
        
        for node_select in most_connected_nodes:
            
            neighbors = list(G.neighbors(node_select))
            if len(neighbors)<=1:
                continue
            #remove_edges = sort_and_select_node(node_list[neighbors],node_curvature[neighbors])[-1]
            
            remove_edges = sort_and_select_node(node_list[neighbors],resistance_matrix.A[node_select][neighbors])[0]
            if(G.has_edge(node_select, remove_edges)):
                G.remove_edge(node_select, remove_edges)
        '''
        for (u, v) in most_pos_edges:
            if(G.has_edge(u, v)):
                G.remove_edge(u, v)
        '''
    edge_index = from_networkx(G).edge_index
    edge_type = torch.zeros(size=(len(G.edges),)).type(torch.LongTensor)
    # edge_type = torch.tensor(edge_type)

    if(debug) : print(f'[INFO] Saving edge_index to {edge_index_filename}')
    with open(edge_index_filename, 'wb') as f:
        torch.save(edge_index, f)

    if(debug) : print(f'[INFO] Saving edge_type to {edge_type_filename}')
    with open(edge_type_filename, 'wb') as f:
        torch.save(edge_type, f)

    return edge_index, edge_type
def resistance_without_curvature(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    batch_add=4,
    batch_remove=2,
    device=None,
    save_dir='rewired_graphs_gerc3',
    dataset_name=None,
    graph_index=0,
    debug=False
):
    print("--------resistance----------")
    # Check if there is a preprocessed graph
    dirname = f'{save_dir}/{dataset_name}'
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    edge_index_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_index_{graph_index}.pt')
    edge_type_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_type_{graph_index}.pt')
    print(edge_index_filename)
    if(os.path.exists(edge_index_filename) and os.path.exists(edge_type_filename)):
        print("--------exisit----------")
        if(debug) : print(f'[INFO] Rewired graph for {loops} iterations, {batch_add} edge additions and {batch_remove} edge removal exists...')
        with open(edge_index_filename, 'rb') as f:
            edge_index = torch.load(f)
        with open(edge_type_filename, 'rb') as f:
            edge_type = torch.load(f)
        return edge_index, edge_type

    # Preprocess data
    G, N, edge_type = _preprocess_data(data)
    
    for _ in range(loops):
        """
        # Compute ORC
        orc = OllivierRicci(G, alpha=0)
        orc.compute_ricci_curvature()
        _C = sorted(orc.G.edges, key=lambda x: orc.G[x[0]][x[1]]['ricciCurvature']['rc_curvature'])
        """
        G = G.to_undirected()
        self_laplacian = nx.laplacian_matrix(G).todense()
        self_pinv = np.linalg.pinv(self_laplacian, hermitian=True)
        pinv_diagonal = np.diag(self_pinv)
        resistance_matrix = np.expand_dims(pinv_diagonal,0) + np.expand_dims(pinv_diagonal,1)  - 2*self_pinv
    # Rewiring begins
        R_list = get_edge_resistance(G.edges,resistance_matrix)
        _C = sort_edge_by_resistance(G.edges,R_list)
        
        # Get top negative and positive curved edges
        most_pos_edges = _C[-batch_remove:]
        most_neg_edges = _C[:batch_add]
       
        '''
        # Add edges
        for (u, v) in most_neg_edges:
            pi = orc.G[u][v]['ricciCurvature']['rc_transport_cost']
            p, q = np.unravel_index(pi.values.argmax(), pi.values.shape)
            p, q = pi.index[p], pi.columns[q]
            if(p != q and not G.has_edge(p, q)):
                G.add_edge(p, q)
        '''
        random_select = False
        # Add edges
        for (u, v) in most_neg_edges:
            '''
            u_neighbors = list(G.neighbors(u)) + [u]
            v_neighbors = list(G.neighbors(v)) + [v]
            if random_select:
                prob_p = softmax(node_curvature[u_neighbors])
                p = np.random.choice(
                    node_curvature[u_neighbors], p=prob_p)
                p = np.where(node_curvature==p)[0][0]
                prob_q = softmax(node_curvature[v_neighbors])
                q = np.random.choice(
                    node_curvature[v_neighbors], p=prob_q)
                q = np.where(node_curvature==q)[0][0]
                if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            else:
                p = np.max(node_curvature[u_neighbors])
                p = np.where(node_curvature==p)[0][0]
                q = np.max(node_curvature[v_neighbors])
                q = np.where(node_curvature==q)[0][0]
                if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            '''
            u_neighbors = list(G.neighbors(u))+[u]
            v_neighbors = list(G.neighbors(v))+[v]
            candidates = []
            candidates_resistance = []
            if len(u_neighbors)==0 or len(v_neighbors)==0:
                continue
            for i in u_neighbors:
                for j in v_neighbors:
                    if (i != j) and (not G.has_edge(i, j)):
                        candidates.append((i, j))
                        candidates_resistance.append(resistance_matrix[i,j])
            if len(candidates)==0:
                continue
            sorted_indices = np.argsort(np.array(candidates_resistance))[-1]
            p,q = candidates[sorted_indices]
            if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            
        # Remove edges
        if G.size()<batch_remove:
            continue
        
       
      
        for (u, v) in most_pos_edges:
            if(G.has_edge(u, v)):
                G.remove_edge(u, v)
  
    edge_index = from_networkx(G).edge_index
    edge_type = torch.zeros(size=(len(G.edges),)).type(torch.LongTensor)
    # edge_type = torch.tensor(edge_type)

    if(debug) : print(f'[INFO] Saving edge_index to {edge_index_filename}')
    with open(edge_index_filename, 'wb') as f:
        torch.save(edge_index, f)

    if(debug) : print(f'[INFO] Saving edge_type to {edge_type_filename}')
    with open(edge_type_filename, 'wb') as f:
        torch.save(edge_type, f)

    return edge_index, edge_type
def gerc3_total_min_resistance(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    batch_add=4,
    batch_remove=2,
    device=None,
    save_dir='rewired_graphs_gerc3',
    dataset_name=None,
    graph_index=0,
    debug=False
):
    print("--------resistance----------")
    # Check if there is a preprocessed graph
    dirname = f'{save_dir}/{dataset_name}'
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    edge_index_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_index_{graph_index}.pt')
    edge_type_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_type_{graph_index}.pt')
    print(edge_index_filename)
    if(os.path.exists(edge_index_filename) and os.path.exists(edge_type_filename)):
        
        print("--------exisit----------")
        G, N, edge_type = _preprocess_data(data)
        R = calculate_ling_resistance(G)
       
        total_R,min_R = total_min_resistance(R,G.edges)
        if(debug) : print(f'[INFO] Rewired graph for {loops} iterations, {batch_add} edge additions and {batch_remove} edge removal exists...')
        with open(edge_index_filename, 'rb') as f:
            edge_index = torch.load(f)
        with open(edge_type_filename, 'rb') as f:
            edge_type = torch.load(f)
        data.edge_index = edge_index
        data.edge_type = edge_type
        G, N, edge_type = _preprocess_data(data)
        R = calculate_ling_resistance(G)
   
        total_R_New,min_R_New = total_min_resistance(R,G.edges)
        total_gain = abs(total_R_New-total_R)/total_R
       
        min_gain = abs(min_R_New-min_R)/min_R
        print(total_gain)
        return total_gain, min_gain

    # Preprocess data
    else:

        return 0,0
def gerc3_resistance_gain(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    batch_add=4,
    batch_remove=2,
    device=None,
    save_dir='rewired_graphs_gerc3',
    dataset_name=None,
    graph_index=0,
    debug=False
):
    print("--------resistance----------")
    # Check if there is a preprocessed graph
    dirname = f'{save_dir}/{dataset_name}'
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    edge_index_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_index_{graph_index}.pt')
    edge_type_filename = os.path.join(dirname, f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_edge_type_{graph_index}.pt')
    print(edge_index_filename)
    if(os.path.exists(edge_index_filename) and os.path.exists(edge_type_filename)):
        print("--------exisit----------")
        if(debug) : print(f'[INFO] Rewired graph for {loops} iterations, {batch_add} edge additions and {batch_remove} edge removal exists...')
        with open(edge_index_filename, 'rb') as f:
            edge_index = torch.load(f)
        with open(edge_type_filename, 'rb') as f:
            edge_type = torch.load(f)
        return edge_index, edge_type

    # Preprocess data
    G, N, edge_type = _preprocess_data(data)
    resistance_matrix = calculate_ling_resistance(G)
    total_R,min_R = total_min_resistance(resistance_matrix,G.edges)
    total_gain = 0
    min_gain = 0
    patience = 0
    past_gain = -99999
    recycle = 0
    while total_gain<0.3 or min_gain<0.3:
        """
        # Compute ORCs
        orc = OllivierRicci(G, alpha=0)
        orc.compute_ricci_curvature()
        _C = sorted(orc.G.edges, key=lambda x: orc.G[x[0]][x[1]]['ricciCurvature']['rc_curvature'])
        """
       
        edge_curvature_vector,node_curvature = resistance_curvature_direct(G,resistance_matrix)
        
        _C = sort_and_select_edge(G.edges,edge_curvature_vector)
        node_list = np.arange(N)
        _N = sort_and_select_node(node_list,node_curvature)
        # Get top negative and positive curved edges
        most_pos_edges = _C[-batch_remove:]
        most_neg_edges = _C[:batch_add]
        most_connected_nodes = _N[-batch_remove:]
        '''
        # Add edges
        for (u, v) in most_neg_edges:
            pi = orc.G[u][v]['ricciCurvature']['rc_transport_cost']
            p, q = np.unravel_index(pi.values.argmax(), pi.values.shape)
            p, q = pi.index[p], pi.columns[q]
            if(p != q and not G.has_edge(p, q)):
                G.add_edge(p, q)
        '''
        random_select = False
        # Add edges
        for (u, v) in most_neg_edges:
            '''
            u_neighbors = list(G.neighbors(u)) + [u]
            v_neighbors = list(G.neighbors(v)) + [v]
            if random_select:
                prob_p = softmax(node_curvature[u_neighbors])
                p = np.random.choice(
                    node_curvature[u_neighbors], p=prob_p)
                p = np.where(node_curvature==p)[0][0]
                prob_q = softmax(node_curvature[v_neighbors])
                q = np.random.choice(
                    node_curvature[v_neighbors], p=prob_q)
                q = np.where(node_curvature==q)[0][0]
                if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            else:
                p = np.max(node_curvature[u_neighbors])
                p = np.where(node_curvature==p)[0][0]
                q = np.max(node_curvature[v_neighbors])
                q = np.where(node_curvature==q)[0][0]
                if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            '''
            u_neighbors = list(G.neighbors(u))+[u]
            v_neighbors = list(G.neighbors(v))+[v]
            candidates = []
            candidates_resistance = []
            if len(u_neighbors)==0 or len(v_neighbors)==0:
                continue
            for i in u_neighbors:
                for j in v_neighbors:
                    if (i != j) and (not G.has_edge(i, j)):
                        candidates.append((i, j))
                        candidates_resistance.append(resistance_matrix[i,j])
            if len(candidates)==0:
                continue
            sorted_indices = np.argsort(np.array(candidates_resistance))[-1]
            p,q = candidates[sorted_indices]
            if(p != q and not G.has_edge(p, q)):
                    G.add_edge(p, q)
            
        # Remove edges
        if G.size()<batch_remove:
            continue
        
        for node_select in most_connected_nodes:
            
            neighbors = list(G.neighbors(node_select))
            if len(neighbors)<=1:
                continue
            #remove_edges = sort_and_select_node(node_list[neighbors],node_curvature[neighbors])[-1]
            
            remove_edges = sort_and_select_node(node_list[neighbors],resistance_matrix.A[node_select][neighbors])[0]
            if(G.has_edge(node_select, remove_edges)):
                G.remove_edge(node_select, remove_edges)
        R = calculate_ling_resistance(G)
        total_R_New,min_R_New = total_min_resistance(R,G.edges)
        min_gain = (min_R_New-min_R)/min_R
        total_gain = (total_R-total_R_New)/total_R
        print("total_gain",total_gain)
        print("min_gain",min_gain)
        '''
        if total_gain<past_gain:
            patience+=1
        if patience>=1:
            break
        '''
        recycle+=1
        if recycle>50:
            break
        past_gain = total_gain
        if total_gain>0.2:
            batch_remove = 10
            batch_add = 0
        '''
        for (u, v) in most_pos_edges:
            if(G.has_edge(u, v)):
                G.remove_edge(u, v)
        '''
    edge_index = from_networkx(G).edge_index
    edge_type = torch.zeros(size=(len(G.edges),)).type(torch.LongTensor)
    # edge_type = torch.tensor(edge_type)

    if(debug) : print(f'[INFO] Saving edge_index to {edge_index_filename}')
    with open(edge_index_filename, 'wb') as f:
        torch.save(edge_index, f)

    if(debug) : print(f'[INFO] Saving edge_type to {edge_type_filename}')
    with open(edge_type_filename, 'wb') as f:
        torch.save(edge_type, f)

    return edge_index, edge_type
'''
if __name__ == '__main__':
    epsilon = 5e-1
    dataset = "ogb"
    if "ogb" not in dataset:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="cora")
    else:
        dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='/PyG/ogbn-arxiv/dataset')
        graph, labels = dataset[0]
        graph = dgl.transforms.add_edges(graph, graph.edges()[1], graph.edges()[0])

        adj = graph.adjacency_matrix_scipy(fmt='csr')
    adj = sys_normalized_adjacency(adj)
    # graph, labels,idx_train, idx_val, idx_test = load_ogb_data(data_use,'/PyG/ogbn-arxiv/dataset')
    print("--------finsh load-------")




    B = spectral_sparsify(adj,epsilon=epsilon, log_every=100,  convergence_after=100, eta=1e-3, max_iters=10000,
                          prevent_vertex_blow_up=False)

    print(f'Sparsified graph has {B.nnz} edges.')
    np.savez_compressed('csr_matrix.npz', data=B.data, indices=B.indices,
                        indptr=B.indptr, shape=B.shape)
    scores = spectral_closeness(sp.csgraph.laplacian(adj), sp.csgraph.laplacian(B))
    rate = np.sum((scores <= epsilon).astype(np.int)) / scores.shape[0] * 100
    print(f'{rate} % of samples deviated at most {epsilon} from the original graph w.r.t. to their laplacians.')

if __name__ == '__main__':
    epsilon = 5e-1
    dataset = "cora"
    if "ogb" not in dataset:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="cora")
    else:
        dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='/PyG/ogbn-arxiv/dataset')
        graph, labels = dataset[0]
        graph = dgl.transforms.add_edges(graph, graph.edges()[1], graph.edges()[0])

        adj = graph.adjacency_matrix_scipy(fmt='csr')
    adj = sys_normalized_adjacency(adj)
    # graph, labels,idx_train, idx_val, idx_test = load_ogb_data(data_use,'/PyG/ogbn-arxiv/dataset')
    print("--------finsh load-------")

    edge_curvature_vector = resistance_curvature(adj, epsilon=epsilon, log_every=100, convergence_after=100, eta=1e-3, max_iters=10000,
                          prevent_vertex_blow_up=False)
    print(edge_curvature_vector)
'''
