import subprocess
import os

# Caminho para o dataset
dataset_path = "Merged01_cleaned.csv"

# Caminho do diretório onde está o script `dpg_custom.py`
script_dir = os.path.abspath("DPG-main/DPG-main")

# Diretório de saída para os resultados
output_dir = "resultados"


# Parâmetros configuráveis
params = {
    "--ds": dataset_path,                      # Dataset
    "--target_column": "Label",                # Coluna alvo
    "--l": "5",                                # Número de árvores no Random Forest
    "--pv": "0.001",                           # Threshold para retenção dos caminhos
    "--t": "2",                                # Precisão decimal das features
    "--model_name": "RandomForestClassifier",  # Modelo escolhido
    "--save_plot_dir": output_dir,             # Diretório onde salvar o plot
    "--attribute": "None",                     # Atributo específico para visualizar (se houver)
}

# Monta o comando dinamicamente com todos os argumentos
command = ["python", "DPG-main/DPG-main/dpg_custom.py"]

for key, value in params.items():
    command.append(str(key))  # Converte a chave para string
    command.append(str(value))  # Converte o valor para string

print("Executando comando:", " ".join(command))  # Debug para verificar o comando

result = subprocess.run(command, capture_output=True, text=True)


# Executa o script
result = subprocess.run(command, capture_output=True, text=True)

# Exibe os resultados
print("="*40)
print("🔹 OUTPUT DO SCRIPT:")
print(result.stdout if result.stdout else "Nenhuma saída gerada.")
print("="*40)

# Exibe erros, se houver
if result.stderr:
    print("⚠️ ERRO ENCONTRADO:")
    print(result.stderr)
