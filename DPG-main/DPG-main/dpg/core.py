import pandas as pd
pd.set_option("display.max_colwidth", 255)
import re
import math
import os
import numpy as np

from collections import defaultdict

import graphviz
import networkx as nx

from tqdm import tqdm

import hashlib
from joblib import Parallel, delayed

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor

def digraph_to_nx(graphviz_graph):
    '''
    This function converts a Graphviz directed graph (DiGraph) to a NetworkX directed graph (DiGraph).
    It also extracts node descriptions and edge weights from the Graphviz graph.

    Args:
    graphviz_graph: The input Graphviz directed graph.

    Returns:
    networkx_graph: The converted NetworkX directed graph.
    nodes_list: A sorted list of nodes with their descriptions.
    '''
    
    # Create an empty directed graph in NetworkX
    networkx_graph = nx.DiGraph()
    
    # Initialize a list to store nodes and a list to store edges with their weights
    nodes_list = []
    edges = []
    weights = {}

    # Extract nodes and edges from the graphviz graph
    for edge in graphviz_graph.body:
        # Check if the line represents an edge (contains '->')
        if "->" in edge:
            # Extract source and destination nodes
            src, dest = edge.split("->")
            src = src.strip()
            dest = dest.split(" [label=")[0].strip()

            # Initialize weight to None
            weight = None
            
            # Extract weight from edge attributes if available
            if "[label=" in edge:
                attr = edge.split("[label=")[1].split("]")[0].split(" ")[0]
                weight = (
                    float(attr)
                    if attr.isdigit() or attr.replace(".", "").isdigit()
                    else None
                )
                weights[(src, dest)] = weight  # Store weight for the edge

            # Add the edge to the list
            edges.append((src, dest))

        # Check if the line represents a node with attributes (contains '[label=')
        if "[label=" in edge:
            id, desc = edge.split("[label=")
            id = id.replace("\t", "")  # Clean up the node ID
            id = id.replace(" ", "")
            desc = desc.split(" fillcolor=")[0]  # Extract node description
            desc = desc.replace('"', "")
            nodes_list.append([id, desc])  # Add node to the list

    # Sort edges and nodes
    edges = sorted(edges)
    nodes_list = sorted(nodes_list, key=lambda x: x[0])

    # Add nodes and edges to the NetworkX graph
    for edge in edges:
        src, dest = edge
        # Add edge with weight if available, else add without weight
        if (src, dest) in weights:
            networkx_graph.add_edge(src, dest, weight=weights[(src, dest)])
        else:
            networkx_graph.add_edge(src, dest)

    # Return the constructed NetworkX graph and the list of nodes
    return networkx_graph, nodes_list


def tracing_ensemble(case_id, sample, ensemble, feature_names, decimal_threshold=2):
    """
    Yields decision paths taken by each tree in an ensemble for a given sample.
    Now uses iteration (not recursion) and yield (not list accumulation).
    """
    is_regressor = isinstance(ensemble, (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor))

    if not isinstance(ensemble, (
        RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,
        RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
    )):
        raise Exception("Ensemble model not recognized!")

    sample = sample.reshape(-1)  # Ensure it's 1D

    for i, tree in enumerate(ensemble.estimators_):
        tree_ = tree.tree_
        node_index = 0
        depth = 0
        prefix = f"sample{case_id}_dt{i}"

        while True:
            left = tree_.children_left[node_index]
            right = tree_.children_right[node_index]
            is_leaf = left == right

            if is_leaf:
                if is_regressor:
                    pred = round(tree_.value[node_index][0][0], 2)
                    yield [prefix, f"Pred {pred}"]
                else:
                    pred_class = tree_.value[node_index].argmax()
                    yield [prefix, f"Class {pred_class}"]
                break

            feature_index = tree_.feature[node_index]
            threshold = round(tree_.threshold[node_index], decimal_threshold)
            feature_name = feature_names[feature_index]
            sample_val = sample[feature_index]

            if sample_val <= threshold:
                condition = f"{feature_name} <= {threshold}"
                node_index = left
            else:
                condition = f"{feature_name} > {threshold}"
                node_index = right

            yield [prefix, condition]
            depth += 1

def filter_log(log, perc_var):
    """
    Low-memory version of variant filtering: avoids large intermediate structures.
    """
    from collections import defaultdict

    # Step 1: Generate variants with minimal memory
    variant_map = defaultdict(list)  # variant -> list of cases
    total_cases = 0

    for case_id, group in log.groupby("case:concept:name", sort=False):
        variant = "|".join(group["concept:name"].values)
        variant_map[variant].append(case_id)
        total_cases += 1

    # Step 2: Filter variants by frequency
    case_ids_to_keep = set()
    min_count = total_cases * perc_var

    for variant, case_ids in variant_map.items():
        if len(case_ids) >= min_count:
            case_ids_to_keep.update(case_ids)

    # Step 3: Filter original log with selected case IDs
    return log[log["case:concept:name"].isin(case_ids_to_keep)].copy()

def discover_dfg(log, n_jobs=1):
    """
    Mines the nodes and edges relationships from an event log and returns a dictionary representing
    the Data Flow Graph (DFG). The DFG shows the frequency of transitions between activities.

    Args:
    log: A pandas DataFrame containing the event log with columns 'case:concept:name' and 'concept:name'.
    n_jobs: Number of parallel jobs to use. Default is -1 (use all available CPUs).

    Returns:
    dfg: A dictionary where keys are tuples representing transitions between activities and values are the counts of those transitions.
    """

    # Pre-group log by case ID (only once!)
    grouped_log = dict(tuple(log.groupby("case:concept:name", sort=False)))

    def process_chunk(chunk):
        chunk_dfg = defaultdict(int)

        for case in chunk:
            trace_df = grouped_log[case]

            # Count transitions
            for a, b in zip(trace_df["concept:name"][:-1], trace_df["concept:name"][1:]):
                chunk_dfg[(a, b)] += 1

        return dict(chunk_dfg)  # Convert back to normal dict for serialization


     

    # Get all unique case names
    cases = log["case:concept:name"].unique()
    print("Remaining paths:", len(cases))
    if len(cases) == 0:
       raise Exception("There is no paths with the current value of perc_var and decimal_threshold! Try one less restrictive.") 

    # If n_jobs is -1, use all available CPUs, otherwise use the provided n_jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count()  # Get the number of available CPU cores
    
    # Ensure n_jobs is at least 1 and no larger than the number of cases
    n_jobs = max(min(n_jobs, len(cases)), 1)  # Ensure n_jobs is within valid range

    # Calculate chunk size, ensure chunk size is at least 1
    chunk_size = max(len(cases) // n_jobs, 1)  # Ensure chunk_size is at least 1
    
    # Split the cases into chunks
    chunks = [cases[i:i + chunk_size] for i in range(0, len(cases), chunk_size)]

    # Process each chunk in parallel
    print("Traversing...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk) for chunk in tqdm(chunks, desc="Chunks")
    )

    # Merge all chunk DFGs into a single DFG dictionary
    print("Aggregating...")
    dfg = {}
    for result in results:
        for key, value in result.items():
            if key in dfg:
                dfg[key] += value  # Aggregate counts for shared transitions
            else:
                dfg[key] = value

    # Return the final DFG dictionary
    return dfg

def generate_dot(dfg):
    """
    Creates a Graphviz directed graph (digraph) from a Data Flow Graph (DFG) dictionary and returns the dot representation.

    Args:
    dfg: A dictionary where keys are tuples representing transitions between activities and values are the counts of those transitions.

    Returns:
    dot: A Graphviz dot object representing the directed graph.
    """

    # Initialize a Graphviz digraph with specified attributes
    dot = graphviz.Digraph(
        "dpg",
        engine="dot",
        graph_attr={
            "bgcolor": "white",
            "rankdir": "R",
            "overlap": "false",
            "fontsize": "20",
        },
        node_attr={"shape": "box"},
    )

    # Keep track of added nodes to avoid duplicates
    added_nodes = set()
    
    # Sort the DFG dictionary by values (transition counts) for deterministic order
    sorted_dict_values = {k: v for k, v in sorted(dfg.items(), key=lambda item: item[1])}

    # Iterate through the sorted DFG dictionary
    for k, v in sorted_dict_values.items():
        
        # Add the source node to the graph if not already added
        if k[0] not in added_nodes:
            dot.node(
                str(int(hashlib.sha1(k[0].encode()).hexdigest(), 16)),
                label=f"{k[0]}",
                style="filled",
                fontsize="20",
                fillcolor="#ffc3c3",
            )
            added_nodes.add(k[0])
        
        # Add the destination node to the graph if not already added
        if k[1] not in added_nodes:
            dot.node(
                str(int(hashlib.sha1(k[1].encode()).hexdigest(), 16)),
                label=f"{k[1]}",
                style="filled",
                fontsize="20",
                fillcolor="#ffc3c3",
            )
            added_nodes.add(k[1])
        
        # Add an edge between the source and destination nodes with the transition count as the label
        dot.edge(
            str(int(hashlib.sha1(k[0].encode()).hexdigest(), 16)),
            str(int(hashlib.sha1(k[1].encode()).hexdigest(), 16)),
            label=str(v),
            penwidth="1",
            fontsize="18"
        )
    
    # Return the Graphviz dot object
    return dot



def calculate_boundaries_old(dict):
    """
    Calculates the boundaries of every feature for every class based on the provided dictionary of predecessors.

    Args:
    dict: A dictionary where keys are class labels and values are lists of predecessor node labels.

    Returns:
    boundaries_class: A dictionary containing the boundaries for each feature of every class.
    """
    # Initialize an empty dictionary to store the boundaries for each class
    boundaries_class = {}

    # Iterate over each class and its corresponding predecessor nodes
    for key, value in dict.items():
        if 'Class' or 'Pred' in key:
            boundaries_class[key] = []
            
            # Extract unique feature names from the predecessor nodes
            key_set = []
            for i in dict[key]:
                key_set.append(str(re.split(' <= | > ', i)[0]))
            key_set = set(key_set)
            
            # Determine boundaries for each unique feature
            for valore_unico in key_set:
                match_list = [math.inf, -math.inf]
                for nodo in dict[key]:
                    if str(re.split(' <= | > ', nodo)[0]) == valore_unico:
                        if '>' in nodo:
                            if float(re.split(' > ', nodo)[1]) < match_list[0]:
                                match_list[0] = float(re.split(' > ', nodo)[1])
                        else:
                            if float(re.split(' <= ', nodo)[1]) > match_list[1]:
                                match_list[1] = float(re.split(' <= ', nodo)[1])

                # Save the boundaries as a string
                alfa = None
                if match_list[0] == math.inf:
                    alfa = str(valore_unico + " <= " + str(match_list[1]))
                elif match_list[1] == -math.inf:
                    alfa = str(valore_unico + " > " + str(match_list[0]))
                else:
                    alfa = str(str(match_list[0]) + ' < ' + valore_unico + ' <= ' + str(match_list[1]))
                boundaries_class[key].append(alfa)

    # Return the dictionary containing class boundaries
    return boundaries_class

def calculate_boundaries_2(class_dict):
    """
    Calculates the boundaries of every feature for every class based on the provided dictionary of predecessors.

    Args:
    class_dict: A dictionary where keys are class labels and values are lists of predecessor node labels.

    Returns:
    boundaries_class: A dictionary containing the boundaries for each feature of every class.
    """
    boundaries_class = {}

    for key, nodes in class_dict.items():
        if 'Class' in key or 'Pred' in key:
            # Initialize boundary dictionary for current class
            feature_bounds = {}

            # Extract conditions from nodes and update boundary values
            for node in nodes:
                parts = re.split(' <= | > ', node)
                feature = parts[0]
                value = float(parts[1])
                condition = '>' in node

                if feature not in feature_bounds:
                    # Initialize with infinities which will be replaced by actual values
                    feature_bounds[feature] = [math.inf, -math.inf]  # [min '>' value, max '<=' value]

                if condition:  # '>' condition
                    if value < feature_bounds[feature][0]:
                        feature_bounds[feature][0] = value
                else:  # '<=' condition
                    if value > feature_bounds[feature][1]:
                        feature_bounds[feature][1] = value

            # Construct boundary descriptions from feature_bounds
            boundaries = []
            for feature, (min_greater, max_lessequal) in feature_bounds.items():
                if min_greater == math.inf:
                    boundary = f"{feature} <= {max_lessequal}"
                elif max_lessequal == -math.inf:
                    boundary = f"{feature} > {min_greater}"
                else:
                    boundary = f"{min_greater} < {feature} <= {max_lessequal}"
                boundaries.append(boundary)

            boundaries_class[key] = boundaries

    return boundaries_class

def calculate_class_boundaries(key, nodes, class_names):
    """
    Calcula os limites de cada feature para uma classe específica e substitui o nome genérico por um nome real.

    Args:
    key: Nome da classe original (ex: 'Class 0').
    nodes: Lista de predecessores dessa classe.
    class_names: Lista com os nomes reais das classes, onde class_names[i] corresponde a 'Class i'.

    Returns:
    (class_name, boundaries): Tupla onde class_name é o nome real e boundaries são as condições associadas.
    """
    # Se a chave for uma classe ('Class X'), troca pelo nome real
    if key.startswith("Class"):
        class_index = int(key.split(" ")[1])  # Pega o número da classe
        class_name = class_names[class_index]  # Substitui pelo nome real
    else:
        class_name = key  # Se não for uma classe, mantém o nome original

    feature_bounds = {}
    boundaries = []

    for node in nodes:
        parts = re.split(' <= | > ', node)
        feature = parts[0]
        value = float(parts[1])
        condition = '>' in node

        if feature not in feature_bounds:
            feature_bounds[feature] = [math.inf, -math.inf]

        if condition:  # Condição '>'
            if value < feature_bounds[feature][0]:
                feature_bounds[feature][0] = value
        else:  # Condição '<='
            if value > feature_bounds[feature][1]:
                feature_bounds[feature][1] = value

    for feature, (min_greater, max_lessequal) in feature_bounds.items():
        if min_greater == math.inf:
            boundary = f"{feature} <= {max_lessequal}"
        elif max_lessequal == -math.inf:
            boundary = f"{feature} > {min_greater}"
        else:
            boundary = f"{min_greater} < {feature} <= {max_lessequal}"
        boundaries.append(boundary)

    return class_name, boundaries  # Retorna a classe já com nome real


def calculate_boundaries(class_dict, class_names):
    """
    Calcula os limites de cada feature para todas as classes, chamando calculate_class_boundaries em paralelo.

    Args:
    class_dict: Dicionário onde as chaves são classes ('Class 0', 'Class 1') e os valores são listas de predecessores.
    class_names: Lista com os nomes reais das classes.

    Returns:
    boundaries_class: Dicionário com os limites para cada classe usando os nomes reais.
    """
    results = Parallel(n_jobs=-1)(
        delayed(calculate_class_boundaries)(key, nodes, class_names) for key, nodes in class_dict.items()
    )
    boundaries_class = dict(results)  # Converte a lista de tuplas para dicionário
    return boundaries_class





def get_dpg_metrics(dpg_model, nodes_list, model):
    """
    Extrai métricas do DPG, incluindo comunidades e limites das classes, com nomes reais.

    Args:
    dpg_model: Grafo NetworkX representando o DPG.
    nodes_list: Lista de nós com identificadores e descrições.
    model: Modelo de ensemble treinado (ex: RandomForestClassifier).

    Returns:
    data: Dicionário contendo as comunidades e os limites das classes.
    """
    np.random.seed(42)
    print("Calculando métricas...")

    # Criar dicionário de nomes dos nós
    diz_nodes = {node[1] if "->" not in node[0] else None: node[0] for node in nodes_list}
    diz_nodes = {k: v for k, v in diz_nodes.items() if k is not None}
    diz_nodes_reversed = {v: k for k, v in diz_nodes.items()}

    # 🔹 Obter os nomes reais das classes diretamente do modelo
    if hasattr(model, "classes_"):
        class_names = list(model.classes_)  # Exemplo: ['Aprovado', 'Reprovado']
    else:
        class_names = []

    # 🔹 Extração de comunidades no grafo
    asyn_lpa_communities = nx.community.asyn_lpa_communities(dpg_model, weight='weight')
    asyn_lpa_communities_stack = [{diz_nodes_reversed[str(node)] for node in community} for community in asyn_lpa_communities]

    # 🔹 Encontrar predecessores das classes
    filtered_nodes = {k: v for k, v in diz_nodes.items() if 'Class' in k or 'Pred' in k}
    predecessors = {k: [] for k in filtered_nodes}
    for key_1, value_1 in filtered_nodes.items():
        try:
            preds = nx.single_source_shortest_path(dpg_model.reverse(), value_1)
            predecessors[key_1] = [k for k, v in diz_nodes.items() if v in preds and k != key_1]
        except nx.NetworkXNoPath:
            continue    

    print("Calculando limites das classes...")
    class_bounds = calculate_boundaries(predecessors, class_names)  

    # Retorno dos dados
    data = {
        "Communities": asyn_lpa_communities_stack,
        "Class Bounds": class_bounds,
    }

    return data


def get_dpg_metrics_to_csv(dpg_model, nodes_list, model, community_file="examples/dpg_communities.csv", bounds_file="examples/dpg_class_bounds.csv"):
    """
    Extrai métricas do DPG e salva os resultados em arquivos CSV para comunidades e limites das classes.

    Args:
    dpg_model: Grafo NetworkX representando o DPG.
    nodes_list: Lista de nós contendo identificadores e descrições.
    model: Modelo de ensemble treinado (ex: RandomForestClassifier).
    community_file: Nome do arquivo CSV para salvar as comunidades.
    bounds_file: Nome do arquivo CSV para salvar os limites das classes.
    """
    np.random.seed(42)

    print("Calculando métricas...")

    # Criar dicionário de nomes dos nós
    diz_nodes = {node[1] if "->" not in node[0] else None: node[0] for node in nodes_list}
    diz_nodes = {k: v for k, v in diz_nodes.items() if k is not None}
    diz_nodes_reversed = {v: k for k, v in diz_nodes.items()}

    # 🔹 Obter os nomes reais das classes diretamente do modelo
    if hasattr(model, "classes_"):
        class_names = list(model.classes_)  # Exemplo: ['Aprovado', 'Reprovado']
    else:
        class_names = []

    # 🔹 Extração de comunidades no grafo
    asyn_lpa_communities = nx.community.asyn_lpa_communities(dpg_model, weight='weight')
    asyn_lpa_communities_stack = [{diz_nodes_reversed[str(node)] for node in community} for community in asyn_lpa_communities]

    # 🔹 Encontrar predecessores das classes
    filtered_nodes = {k: v for k, v in diz_nodes.items() if 'Class' in k or 'Pred' in k}
    predecessors = {k: [] for k in filtered_nodes}
    for key_1, value_1 in filtered_nodes.items():
        try:
            preds = nx.single_source_shortest_path(dpg_model.reverse(), value_1)
            predecessors[key_1] = [k for k, v in diz_nodes.items() if v in preds and k != key_1]
        except nx.NetworkXNoPath:
            continue    

    print("Calculando limites das classes...")
    class_bounds = calculate_boundaries(predecessors, class_names)  

    # 🔹 Salvar dados das comunidades
    community_data = []
    for i, community in enumerate(asyn_lpa_communities_stack):
        for node in community:
            community_data.append([f"Community_{i+1}", node])

    df_communities = pd.DataFrame(community_data, columns=["Community", "Node"])
    df_communities.to_csv(community_file, index=False)
    print(f"DPG communities saved to {community_file}")

    # 🔹 Salvar limites das classes
    bounds_data = []
    for key, bounds in class_bounds.items():
        for constraint in bounds:
            bounds_data.append([key, constraint])

    df_bounds = pd.DataFrame(bounds_data, columns=["Class", "Constraint"])
    df_bounds.to_csv(bounds_file, index=False)
    print(f"DPG class bounds saved to {bounds_file}")

    return df_communities, df_bounds



def get_dpg_node_metrics(dpg_model, nodes_list):
    """
    Extracts metrics from the nodes of a DPG model.

    Args:
    dpg_model: A NetworkX graph representing the DPG.
    nodes_list: A list of nodes where each node is a tuple. The first element is the node identifier and the second is the node label.

    Returns:
    df: A pandas DataFrame containing the metrics for each node in the DPG.
    """
    
    # Calculate the degree of each node
    degree = dict(nx.degree(dpg_model))
    # Calculate the in-degree (number of incoming edges) for each node
    in_nodes = {node: dpg_model.in_degree(node) for node in list(dpg_model.nodes())}
    # Calculate the out-degree (number of outgoing edges) for each node
    out_nodes = {node: dpg_model.out_degree(node) for node in list(dpg_model.nodes())}
    # Calculate the betweenness centrality for each node
    betweenness_centrality = nx.betweenness_centrality(dpg_model, weight='weight')
    # Calculate the local reaching centrality for each node
    local_reaching_centrality = {node: nx.local_reaching_centrality(dpg_model, node, weight='weight') for node in list(dpg_model.nodes())}
    
    # Create a dictionary to store the node metrics
    data_node = {
        "Node": list(dpg_model.nodes()),
        "Degree": list(degree.values()),                               # Total degree (in-degree + out-degree)
        "In degree nodes": list(in_nodes.values()),                    # Number of incoming edges
        "Out degree nodes": list(out_nodes.values()),                  # Number of outgoing edges
        "Betweenness centrality": list(betweenness_centrality.values()),   # Betweenness centrality (useful for identifying bottlenecks)
        "Local reaching centrality": list(local_reaching_centrality.values()),  # Local reaching centrality (useful for feature importance)
    }

    # Merge the node metrics with the node labels
    # Assuming data_node and nodes_list are your input data sets
    df_data_node = pd.DataFrame(data_node).set_index('Node')
    df_nodes_list = pd.DataFrame(nodes_list, columns=["Node", "Label"]).set_index('Node')
    df = pd.concat([df_data_node, df_nodes_list], axis=1, join='inner').reset_index()
    
    # Return the resulting DataFrame
    return df



def get_dpg(X_train, feature_names, model, perc_var, decimal_threshold, n_jobs=-1):
    """
    Generates a DPG from training data and a random forest model.

    Args:
    X_train: A numpy array or similar structure containing the training data samples.
    feature_names: A list of feature names corresponding to the columns in X_train.
    model: A trained random forest model.
    perc_var: A float representing the minimum percentage of total traces a variant must have to be kept. 
    In practical terms, consider a problem with 1,000 paths (learner x training samples), if the minimum percentage threshold is set at 5% (0.05), 
    only those variants that appear in at least 50 of these paths (5% of 1,000) would be retained for further analysis. 
    This helps in reducing the noise and complexity of the data, allowing analysts to concentrate on more common and potentially significant patterns.
    decimal_threshold: The number of decimal places to which thresholds are rounded.
    n_jobs: Number of parallel jobs to run. Default is -1 (use all available CPUs).

    Returns:
    dot: A Graphviz Digraph object representing the DPG.
    """

    print("\nStarting DPG extraction *****************************************")
    print("Model Class:", model.__class__.__name__)
    print("Model Class Module:", model.__class__.__module__)
    print("Model Estimators: ", len(model.estimators_))
    print("Model Params: ", model.get_params())
    print("*****************************************************************")

    def process_sample(i, sample):
        """Process a single sample."""
        return list(tracing_ensemble(i, sample, model, feature_names, decimal_threshold))

    print('Tracing ensemble...')
    log = Parallel(n_jobs=n_jobs)(
        delayed(process_sample)(i, sample) for i, sample in tqdm(list(enumerate(X_train)), total=len(X_train))
    )

    # Flatten the list of lists
    log = [item for sublist in log for item in sublist]
    log_df = pd.DataFrame(log, columns=["case:concept:name", "concept:name"])
    print(f'Total of paths: {len(log_df["case:concept:name"].unique())}')
    
    print(f'Filtering structure... (perc_var={perc_var})')
    # Filter the log based on the variant percentage if specified
    filtered_log = log_df
    if perc_var > 0:
        filtered_log = filter_log(log_df, perc_var)
    
    print('Building DPG...')
    # Discover the Data Flow Graph (DFG) from the filtered log
    dfg = discover_dfg(filtered_log)

    print('Extracting graph...')
    # Create a Graphviz Digraph object from the DFG
    dot = generate_dot(dfg)
    return dot