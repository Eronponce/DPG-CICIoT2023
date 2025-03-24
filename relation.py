import pandas as pd
import numpy as np
import zipfile
import os
from scipy.stats import spearmanr, chi2_contingency

# Caminho para seu arquivo ZIP (atualize conforme necessário)
zip_path = 'examples.zip'
extracted_path = 'examples_final/'

# Extrair arquivos do ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# Diretório contendo as execuções
base_dir = os.path.join(extracted_path, 'examples')

num_execucoes = 10
importances_rf = []
metrics_lrc = []

# Processa as 10 execuções
for i in range(1, num_execucoes + 1):
    dir_exec = os.path.join(base_dir, f"resultados_{i}")

    # Carrega RF Importance
    rf_df = pd.read_csv(os.path.join(dir_exec, "custom_l5_importance_rf.csv"))
    importances_rf.append(rf_df.set_index("Feature")["Importance"])

    # Carrega e agrega Local Reaching Centrality (LRC)
    lrc_df = pd.read_csv(os.path.join(dir_exec, "custom_l5_pv0.001_t2_node_metrics.csv"))
    lrc_df["Feature"] = lrc_df["Label"].apply(lambda x: x.split()[0])
    lrc_agg = lrc_df.groupby("Feature")["Local reaching centrality"].sum()
    metrics_lrc.append(lrc_agg)

# Consolidar rankings das métricas
rf_df_final = pd.concat(importances_rf, axis=1).mean(axis=1).rank(ascending=False)
lrc_df_final = pd.concat(metrics_lrc, axis=1).mean(axis=1).rank(ascending=False)

# Cria DataFrame combinado dos rankings
rankings_df = pd.DataFrame({
    "Rank_RF": rf_df_final,
    "Rank_LRC": lrc_df_final
}).dropna()

# Calcula o ranking médio combinado
rankings_df['Rank_Mean'] = rankings_df.mean(axis=1)
rankings_df.sort_values('Rank_Mean', inplace=True)

# Correlação Spearman
spearman_corr, spearman_p = spearmanr(rankings_df["Rank_RF"], rankings_df["Rank_LRC"])

# Codifica rankings em categorias para o teste Qui-quadrado
rankings_df['Rank_RF_cat'] = pd.qcut(rankings_df['Rank_RF'], q=3, labels=["Alta", "Média", "Baixa"])
rankings_df['Rank_LRC_cat'] = pd.qcut(rankings_df['Rank_LRC'], q=3, labels=["Alta", "Média", "Baixa"])

# Cria tabela de contingência
contingency_table = pd.crosstab(rankings_df['Rank_RF_cat'], rankings_df['Rank_LRC_cat'])

# Realiza o teste Qui-quadrado
chi2, chi_p, dof, expected = chi2_contingency(contingency_table)

# Salva resultados em arquivos CSV
rankings_df.to_csv('ranking_combinado.csv', index=True)
contingency_table.to_csv('tabela_contingencia.csv', index=True)

# Salva também os resultados dos testes em um CSV separado
results_summary = pd.DataFrame({
    'Teste': ['Correlação de Spearman', 'Qui-quadrado'],
    'Estatística': [spearman_corr, chi2],
    'p-valor': [spearman_p, chi_p]
})

results_summary.to_csv('resultados_estatisticos.csv', index=False)
