import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import time
from qiskit_aer import Aer
from qiskit_algorithms import SamplingVQE
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE
from qiskit_aer.primitives import Estimator
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import BackendSampler
from qiskit_algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSampler



# --- Graph Generation ---
def create_graph(graph_type, n, p=0.5, m=2, k=4):
    if graph_type == "Erdős-Rényi":
        return nx.erdos_renyi_graph(n, p)
    elif graph_type == "Barabási-Albert":
        return nx.barabasi_albert_graph(n, m)
    elif graph_type == "Watts-Strogatz":
        return nx.watts_strogatz_graph(n, k, p)
    else:
        return nx.gnm_random_graph(n, n)
    

# --- VQE Solver ---
def solve_maxcut_vqe(G):
    maxcut = Maxcut(G)
    qp = maxcut.to_quadratic_program()

    ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
    sampler = Sampler()
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=COBYLA())

    # Directly use SamplingVQE here
    optimizer = MinimumEigenOptimizer(vqe)

    start = time.time()
    result = optimizer.solve(qp)
    end = time.time()

    cut = maxcut.interpret(result)
    return cut, result.fval, end - start    



# --- QAOA Solver ---
def solve_maxcut_qaoa(G, p_level=1):
    maxcut = Maxcut(G)
    qp = maxcut.to_quadratic_program()
    backend = Aer.get_backend('aer_simulator')
    sampler = BackendSampler(backend)
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p_level)
    optimizer = MinimumEigenOptimizer(qaoa)

    start = time.time()
    result = optimizer.solve(qp)
    end = time.time()

    cut = maxcut.interpret(result)
    return cut, result.fval, end - start



# --- Brute-Force Max-Cut (for small graphs) ---
def solve_maxcut_brute_force(G):
    nodes = list(G.nodes)
    max_cut_value = 0
    best_cut = None

    start = time.time()
    for bits in itertools.product([0, 1], repeat=len(nodes)):
        cut_value = 0
        for u, v in G.edges:
            if bits[u] != bits[v]:
                cut_value += 1
        if cut_value > max_cut_value:
            max_cut_value = cut_value
            best_cut = list(bits)
    end = time.time()
    return best_cut, max_cut_value, end - start

# --- Greedy Heuristic Max-Cut ---
def solve_maxcut_greedy(G):
    nodes = list(G.nodes)
    part = {node: 0 for node in nodes}
    for node in nodes:
        cut0 = sum(1 for nbr in G.neighbors(node) if part[nbr] == 0)
        cut1 = sum(1 for nbr in G.neighbors(node) if part[nbr] == 1)
        part[node] = 1 if cut0 > cut1 else 0
    cut_value = sum(1 for u, v in G.edges if part[u] != part[v])
    cut = [part[i] for i in nodes]
    return cut, cut_value, 0  # negligible time


# --- Cut Visualizer ---
def draw_cut(G, cut):
    # Ensure cut is the same length as nodes
    if len(cut) < len(G.nodes):
        cut += [0] * (len(G.nodes) - len(cut))
    
    pos = nx.spring_layout(G, seed=42)
    node_colors = ['red' if cut[node] == 0 else 'blue' for node in G.nodes]

    # Determine edge colors
    cut_edges = []
    same_group_edges = []
    for u, v in G.edges:
        if cut[u] != cut[v]:
            cut_edges.append((u, v))
        else:
            same_group_edges.append((u, v))

    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=same_group_edges, edge_color='gray', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='green', style='dashed', ax=ax)

    ax.set_title("Max-Cut Partition Visualization")
    ax.axis('off')
    return fig

# --- Streamlit UI ---
st.title("Max-Cut Solver")
st.markdown("Interactively solve the Max-Cut problem using **QAOA** or **Other available algorithms** for both classical and quantum approaches")
st.markdown("WARNING: Run times may greatly differ depending on graph type, size and Algorithm chosen")

# Sidebar Controls
graph_type = st.sidebar.selectbox("Graph Type", ["Erdős-Rényi", "Barabási-Albert", "Watts-Strogatz"])
n_nodes = st.sidebar.slider("Number of Nodes", 4, 30, 10)
algorithm = st.sidebar.selectbox("Algorithm", ["QAOA", "VQE", "Greedy", "Brute Force", "Compare All"])
p = st.sidebar.slider("Edge Probability (p)", 0.1, 1.0, 0.5)
m = st.sidebar.slider("Edges to Attach (m, BA)", 1, 5, 2)
k = st.sidebar.slider("Nearest Neighbors (k, WS)", 2, 6, 4)
qaoa_depth = st.sidebar.slider("QAOA Depth (p)", 1, 5, 1)

if st.button("Solve Max-Cut"):
    G = create_graph(graph_type, n_nodes, p=p, m=m, k=k)

    if algorithm in ["QAOA", "Compare All"]:
        st.subheader("QAOA Result")
        q_cut, q_val, q_time = solve_maxcut_qaoa(G, qaoa_depth)
        st.write(f"Cut Value: {q_val:.2f} | Time: {q_time:.3f}s")
        st.pyplot(draw_cut(G, q_cut))

    if algorithm in ["VQE", "Compare All"]:
        st.subheader("VQE Result")
        v_cut, v_val, v_time = solve_maxcut_vqe(G)
        st.write(f"Cut Value: {v_val:.2f} | Time: {v_time:.3f}s")
        st.pyplot(draw_cut(G, v_cut))

    if algorithm in ["Greedy", "Compare All"]:
        st.subheader("Greedy Heuristic Result")
        g_cut, g_val, _ = solve_maxcut_greedy(G)
        st.write(f"Cut Value: {g_val}")
        st.pyplot(draw_cut(G, g_cut))

    if algorithm in ["Brute Force", "Compare All"] and n_nodes <= 15:
        st.subheader("Brute Force Result")
        bf_cut, bf_val, bf_time = solve_maxcut_brute_force(G)
        st.write(f"Cut Value: {bf_val} | Time: {bf_time:.3f}s")
        st.pyplot(draw_cut(G, bf_cut))
    else: 
        print("That graph is too big for a brute force method, must be 15 nodes or smaller")


