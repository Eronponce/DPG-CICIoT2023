import subprocess
import os
import random
import shutil

dataset_path = os.path.abspath("Merged01.csv")
num_execucoes = 10
examples_dir = "examples"  # Diretório onde os arquivos são gerados

for i in range(1, num_execucoes + 1):
    seed = random.randint(0, 999999)

    params = {
        "--ds": dataset_path,
        "--target_column": "Label",
        "--l": "5",
        "--pv": "0.001",
        "--t": "2",
        "--model_name": "RandomForestClassifier",
        "--save_plot_dir": examples_dir,
        "--attribute": "None",
        "--n_jobs": "6",
        "--perc_dataset": "0.01",
        "--seed": str(seed),
    }

    command = ["python", "DPG-main/DPG-main/dpg_custom.py"]
    
    # Adiciona os parâmetros ao comando
    for key, value in params.items():
        command.append(str(key))
        command.append(str(value))

    # Adiciona a flag --importance sem valor associado
    command.append("--importance")

    subprocess.run(command)

    # Criar novo diretório para armazenar os arquivos processados
    output_dir = os.path.join("examples", f"resultados_{i}")
    os.makedirs(output_dir, exist_ok=True)

    # Mover arquivos do diretório Examples para o novo diretório
    for file_name in os.listdir(examples_dir):
        file_path = os.path.join(examples_dir, file_name)
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join(output_dir, file_name))

    print(f"Execução {i} concluída. Resultados movidos para {output_dir}")
