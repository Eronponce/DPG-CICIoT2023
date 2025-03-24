import os
import argparse
import dpg.sklearn_custom_dpg as test
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="Basic dataset to be analyzed")
    parser.add_argument("--target_column", type=str, help="Name of the column to be used as the target variable")
    parser.add_argument("--l", type=int, default=5, help="Number of learners for the Random Forest")
    parser.add_argument("--pv", type=float, default=0.001, help="Threshold value for path selection in DPG")
    parser.add_argument("--t", type=int, default=2, help="Decimal precision of each feature")
    parser.add_argument("--model_name", type=str, default="RandomForestClassifier", help="Chosen ensemble model")
    parser.add_argument("--dir", type=str, default="examples/", help="Directory to save results")
    parser.add_argument("--plot", action='store_true', help="Plot the DPG, add the argument to use it as True")
    parser.add_argument("--save_plot_dir", type=str, default="examples/", help="Directory to save the plot image")
    parser.add_argument("--attribute", type=str, default=None, help="A specific node attribute to visualize")
    parser.add_argument("--communities", action='store_true', help="Boolean indicating whether to visualize communities")
    parser.add_argument("--class_flag", action='store_true', help="Boolean indicating whether to highlight class nodes")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs to run in parallel")
    parser.add_argument("--perc_dataset", type=float, default=1.0, help="Percentage of the dataset to be used")
    parser.add_argument("--importance", action='store_true', help="Calculate feature importance (Random Forest & LCR)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for dataset sampling")

    args = parser.parse_args()

    df, df_dpg_metrics, df_rf_importance = test.test_base_sklearn(
        datasets=args.ds,
        target_column=args.target_column,
        n_learners=args.l, 
        perc_var=args.pv, 
        decimal_threshold=args.t,
        model_name=args.model_name,
        file_name=os.path.join(args.dir, f'custom_l{args.l}_pv{args.pv}_t{args.t}_stats.txt'), 
        plot=args.plot, 
        save_plot_dir=args.save_plot_dir, 
        attribute=args.attribute, 
        communities=args.communities, 
        class_flag=args.class_flag,
        n_jobs=args.n_jobs,
        perc_dataset=args.perc_dataset,
        importance=args.importance,
        random_seed=args.seed  # Passando o seed para garantir reprodutibilidade
    )

    df.to_csv(os.path.join(args.dir, f'custom_l{args.l}_pv{args.pv}_t{args.t}_node_metrics.csv'), encoding='utf-8')

    with open(os.path.join(args.dir, f'custom_l{args.l}_pv{args.pv}_t{args.t}_dpg_metrics.txt'), 'w') as f:
        for key, value in df_dpg_metrics.items():
            f.write(f"{key}: {value}\n")

    if args.importance:
        df_rf_importance.to_csv(os.path.join(args.dir, f'custom_l{args.l}_importance_rf.csv'), encoding='utf-8', index=False)
