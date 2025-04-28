import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from .core import digraph_to_nx, get_dpg, get_dpg_node_metrics, get_dpg_metrics, get_dpg_metrics_to_csv
from .visualizer import plot_dpg
from sklearn.metrics import accuracy_score  # Adicionei a importação do accuracy_score

def select_custom_dataset(path, target_column, perc_dataset=1.0, random_seed=None):
    """
    Loads a dataset from a CSV file, samples a percentage, and prepares it for modeling.

    Args:
        path (str): Path to the CSV dataset.
        target_column (str): Name of the target variable column.
        perc_dataset (float, optional): Fraction of dataset to use (default is 1.0).
        random_seed (int, optional): Seed for reproducibility. If None, defaults to 42.

    Returns:
        tuple: (data, features, target)
    """
    df = pd.read_csv(path, sep=',')
    
    # Se perc_dataset < 1.0, fazer amostragem aleatória
    if perc_dataset < 1.0:
        if random_seed is None:
            random_seed = 42  # Valor padrão caso não seja fornecido
        df = df.sample(frac=perc_dataset, random_state=random_seed).reset_index(drop=True)
    
    target = df.pop(target_column).values
    data = df.values
    features = df.columns.to_numpy()

    return data, features, target

def test_base_sklearn(datasets, target_column, n_learners, perc_var, decimal_threshold, model_name='RandomForestClassifier',
                      file_name=None, plot=False, save_plot_dir="examples/", attribute=None, communities=False, 
                      class_flag=False, n_jobs=-1, perc_dataset=1.0, importance=False, random_seed=42, n_estimators=100, max_depth=None, 
                      min_samples_split=2, min_samples_leaf=1,balanced=False): 
    
    if file_name:
        output_dir = os.path.dirname(file_name)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if save_plot_dir and not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)

    # Load dataset com a seed definida
    data, features, target = select_custom_dataset(datasets, target_column=target_column, perc_dataset=perc_dataset, random_seed=random_seed)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=random_seed)
    
    # Escolher o modelo com base no argumento passado
    if model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=n_learners,  # usado como fallback, mas será sobrescrito logo abaixo
            random_state=random_seed,
            class_weight='balanced' if balanced else None
        )
        # Sobrescreve com os hiperparâmetros passados
        model.set_params(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
    elif model_name == 'ExtraTreesClassifier':
        model = ExtraTreesClassifier(n_estimators=n_learners, random_state=random_seed)
    elif model_name == 'AdaBoostClassifier':
        model = AdaBoostClassifier(n_estimators=n_learners, random_state=random_seed)
    elif model_name == 'BaggingClassifier':
        model = BaggingClassifier(n_estimators=n_learners, random_state=random_seed)
    else:
        raise Exception("The selected model is not currently available.")
            
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  # Calculando a acurácia
    df_rf_importance = None
    if importance and hasattr(model, "feature_importances_"):
        df_rf_importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        df_rf_importance = df_rf_importance.sort_values(by="Importance", ascending=False)
    
    dot = get_dpg(X_train, features, model, perc_var, decimal_threshold, n_jobs=n_jobs)
    dpg_model, nodes_list = digraph_to_nx(dot)

    df_dpg = get_dpg_metrics(dpg_model, nodes_list, model)
    df = get_dpg_node_metrics(dpg_model, nodes_list)
    get_dpg_metrics_to_csv(dpg_model, nodes_list, model)

    if plot:
        plot_dpg("plot_name", dot, df, df_dpg, save_dir=save_plot_dir, attribute=attribute, communities=communities, class_flag=class_flag)

    return df, df_dpg, df_rf_importance, accuracy  # Retornando a acurácia também
