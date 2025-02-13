import pandas as pd
import numpy as np
import ntpath
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import is_classifier, is_regressor

from .core import digraph_to_nx, get_dpg, get_dpg_node_metrics, get_dpg_metrics, get_dpg_metrics_to_csv
from .visualizer import plot_dpg


def select_custom_dataset(path, target_column, perc_dataset=1.0, random_state=42):
    

    """
    Loads a custom dataset from a CSV file, optionally samples a percentage, 
    separates the target column, and prepares the data for modeling.

    Args:
        path (str): File path to the CSV dataset.
        target_column (str): Name of the target variable column.
        perc_dataset (float, optional): Fraction of dataset to use (default is 1.0 for full dataset).
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (data, features, target)
            - data (numpy.ndarray): Feature data.
            - features (numpy.ndarray): Feature names.
            - target (numpy.ndarray): Target variable.
    """
    
    # Load dataset
    df = pd.read_csv(path, sep=',')
    
    # Sample a percentage of the dataset (if less than 100%)
    if perc_dataset < 1.0:
        df = df.sample(frac=perc_dataset, random_state=random_state).reset_index(drop=True)
    
    # Extract target variable
    target = df.pop(target_column).values  # Removes target column and converts to NumPy array
    
    # Extract feature data and feature names
    data = df.values  # Converts remaining dataframe to numpy array efficiently
    features = df.columns.to_numpy()  # Extracts feature names directly

    return data, features, target



def test_base_sklearn(datasets, target_column, n_learners, perc_var, decimal_threshold, model_name='RandomForestClassifier',
                      file_name=None, plot=False, save_plot_dir="examples/", attribute=None, communities=False, 
                      class_flag=False, n_jobs=-1, perc_dataset=1.0, importance=False):
    
    if file_name:
        output_dir = os.path.dirname(file_name)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if save_plot_dir and not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)

    # Load dataset
    data, features, target = select_custom_dataset(datasets, target_column=target_column, perc_dataset=perc_dataset)
    
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    
    # Train model
    if model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=n_learners, random_state=42)
    elif model_name == 'ExtraTreesClassifier':
        model = ExtraTreesClassifier(n_estimators=n_learners, random_state=42)
    elif model_name == 'AdaBoostClassifier':
        model = AdaBoostClassifier(n_estimators=n_learners, random_state=42)
    elif model_name == 'BaggingClassifier':
        model = BaggingClassifier(n_estimators=n_learners, random_state=42)
    else:
        raise Exception("The selected model is not currently available.")
            
    model.fit(X_train, y_train)
    
    # Feature Importance from Random Forest
    df_rf_importance = None
    if importance and hasattr(model, "feature_importances_"):
        df_rf_importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        df_rf_importance = df_rf_importance.sort_values(by="Importance", ascending=False)
    
    # Extract DPG
    dot = get_dpg(X_train, features, model, perc_var, decimal_threshold, n_jobs=n_jobs)
    dpg_model, nodes_list = digraph_to_nx(dot)

    df_dpg = get_dpg_metrics(dpg_model, nodes_list)
    df = get_dpg_node_metrics(dpg_model, nodes_list)
    get_dpg_metrics_to_csv(dpg_model, nodes_list)
    if plot:
        plot_dpg("plot_name", dot, df, df_dpg, save_dir=save_plot_dir, attribute=attribute, communities=communities, class_flag=class_flag)

    return df, df_dpg, df_rf_importance