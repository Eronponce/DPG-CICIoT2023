import pandas as pd

# Carregar o dataset
input_file = "Merged01.csv"  # Substitua pelo nome do seu arquivo
output_file = "subconjunto_1_porcento.csv"

# Definir a seed para reprodutibilidade
seed = 42

# Ler o arquivo CSV
df = pd.read_csv(input_file)

# Amostrar 1% do dataset
df_sample = df.sample(frac=0.01, random_state=seed)

# Salvar o subconjunto em um novo CSV
df_sample.to_csv(output_file, index=False)

print(f"Subconjunto salvo em: {output_file}")
