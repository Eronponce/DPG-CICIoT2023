import subprocess
import os

dataset_path = "Merged01.csv"

script_dir = os.path.abspath("DPG-main/DPG-main")

# Número de execuções
num_execucoes = 1

# Loop para executar o script 10 vezes
for i in range(1, num_execucoes + 1):
    # Diretório específico para cada execução
    output_dir = f"resultados_{i}"
    
    # Cria o diretório se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Parâmetros configuráveis
    params = {
        "--ds": dataset_path,                      # Dataset
        "--target_column": "Label",                # Coluna alvo
        "--l": "5",                                # Número de árvores no Random Forest
        "--pv": "0.01",                           # Threshold para retenção dos caminhos
        "--t": "2",                                # Precisão decimal das features
        "--model_name": "RandomForestClassifier",  # Modelo escolhido
        "--save_plot_dir": output_dir,             # Diretório diferente a cada execução
        "--attribute": "None",  
        "--n_jobs": "1",
        "--perc_dataset":"0.01"    

    }

    # Monta o comando dinamicamente com todos os argumentos
    command = ["python", "DPG-main/DPG-main/dpg_custom.py"]

    for key, value in params.items():
        command.append(str(key))
        command.append(str(value))

    print(f"🔄 Executando tentativa {i}/{num_execucoes}...")
    print("Comando:", " ".join(command))

    # Executa o script
    result = subprocess.run(command, capture_output=True, text=True)

    # Exibe os resultados
    print("=" * 40)
    print(f"🔹 OUTPUT DA EXECUÇÃO {i}:")
    print(result.stdout if result.stdout else "Nenhuma saída gerada.")
    print("=" * 40)

    # Exibe erros, se houver
    if result.stderr:
        print(f"⚠️ ERRO NA EXECUÇÃO {i}:")
        print(result.stderr)
