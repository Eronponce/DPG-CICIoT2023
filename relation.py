import pandas as pd
import os
from scipy.stats import spearmanr, chi2_contingency

base_dir = 'examples'
num_execucoes = 5  # alinhado com o número real de execuções

importances_rf = []
metrics_lrc = []
accuracies = {"Estilo 1": [], "Estilo 2": [], "Estilo 3": []}

styles_params = [
    ("custom_l10", "pv0.001_t1"),
    ("custom_l3", "pv0.001_t1"),
    ("custom_l5", "pv0.001_t2")
]

# Processa cada execução
for i in range(1, num_execucoes + 1):
    dir_exec = os.path.join(base_dir, f"resultados_{i}")
    estilo_idx = (i - 1) % 3
    estilo = f"Estilo {estilo_idx + 1}"
    prefix, suffix = styles_params[estilo_idx]

    # Carregar RF Importance
    rf_df = pd.read_csv(os.path.join(dir_exec, f"{prefix}_importance_rf.csv"))
    importances_rf.append(rf_df.set_index("Feature")["Importance"])

    # Carregar LRC metrics
    lrc_df = pd.read_csv(os.path.join(dir_exec, f"{prefix}_{suffix}_node_metrics.csv"))
    lrc_df["Feature"] = lrc_df["Label"].apply(lambda x: x.split()[0])
    lrc_agg = lrc_df.groupby("Feature")["Local reaching centrality"].sum()
    metrics_lrc.append(lrc_agg)

    # Carregar acurácia
    accuracy_file_path = os.path.join(dir_exec, f"{prefix}_{suffix}_accuracy.txt")
    if os.path.exists(accuracy_file_path):
        with open(accuracy_file_path, 'r') as file:
            accuracy_line = file.readline()
            accuracy = float(accuracy_line.split(":")[1].strip())
            accuracies[estilo].append(accuracy)

# Ranking combinado
rf_df_final = pd.concat(importances_rf, axis=1).mean(axis=1).rank(ascending=False)
lrc_df_final = pd.concat(metrics_lrc, axis=1).mean(axis=1).rank(ascending=False)

rankings_df = pd.DataFrame({
    "Rank_RF": rf_df_final,
    "Rank_LRC": lrc_df_final
}).dropna()
rankings_df['Rank_Mean'] = rankings_df.mean(axis=1)
rankings_df.sort_values('Rank_Mean', inplace=True)

# Teste Spearman
spearman_corr, spearman_p = spearmanr(rankings_df["Rank_RF"], rankings_df["Rank_LRC"])

# Teste Qui-quadrado
rankings_df['Rank_RF_cat'] = pd.qcut(rankings_df['Rank_RF'], q=3, labels=["Alta", "Média", "Baixa"])
rankings_df['Rank_LRC_cat'] = pd.qcut(rankings_df['Rank_LRC'], q=3, labels=["Alta", "Média", "Baixa"])
contingency_table = pd.crosstab(rankings_df['Rank_RF_cat'], rankings_df['Rank_LRC_cat'])
chi2, chi_p, _, _ = chi2_contingency(contingency_table)

# Salvar resultados em arquivos CSV
rankings_df.to_csv('ranking_combinado.csv', index=True)
contingency_table.to_csv('tabela_contingencia.csv', index=True)

# Salvar resultados dos testes estatísticos em arquivo CSV separado
results_summary = pd.DataFrame({
    'Teste': ['Correlação de Spearman', 'Qui-quadrado'],
    'Estatística': [spearman_corr, chi2],
    'p-valor': [spearman_p, chi_p]
})
results_summary.to_csv('resultados_estatisticos.csv', index=False)

# Salvar comparação das acurácias em arquivo CSV
accuracies_df = pd.DataFrame({
    estilo: [sum(valores) / len(valores) if valores else 0] for estilo, valores in accuracies.items()
})
accuracies_df.to_csv('comparacao_acuracias.csv', index=False)
