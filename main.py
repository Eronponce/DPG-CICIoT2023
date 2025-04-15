import subprocess
import shutil
import os
import pandas as pd

# Caminhos
dataset_path = "Merged01_6label_no_web_brute.csv"
examples_dir = "examples"

# Parâmetros para execução com modelo treinado e hiperparâmetros manuais
params = {
    "--ds": dataset_path,
    "--target_column": "Main_Label",
    "--pv": "0.01",
    "--t": "2",
    "--save_plot_dir": examples_dir,
    "--attribute": "None",
    "--n_jobs": "1",
    "--perc_dataset": "1",
    "--model_name": "RandomForestClassifier",
    "--n_estimators": "78",
    "--max_depth": "16",
    "--min_samples_split": "2",
    "--min_samples_leaf": "2",
    "--importance": "",


}

# Loop de execuções (exemplo com 1 execução)
for i in range(1):
    command = ["python", "DPG-main/DPG-main/dpg_custom.py"]

    for key, value in params.items():
        command.append(str(key))
        if value != "":
            command.append(str(value))

    print("Executando:", " ".join(command))
    subprocess.run(command)

    # Salvar resultados da execução
    output_dir = os.path.join("examples", f"dpg_model_exec_{i}")
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(examples_dir):
        file_path = os.path.join(examples_dir, file_name)
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join(output_dir, file_name))
