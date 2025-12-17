
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


df = pd.read_csv("player_features_updated.csv", sep=";", decimal=",")

corr_vars = [
    "std_morality",
    "final_morality",
    "number_puzzles_started",
    "number_puzzles_succeeded",
    "%_moral_decisions",
    "state_entropy",
    "%_exploration",
    "pct_time_in_hotspots",
    "number_hotspots",
    "average_time_spent_puzzles",
    "%_puzzle",
    "time_complete_game"
]


corr = df[corr_vars].corr(method="pearson")
corr_matrix = corr.values

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(corr_matrix)

for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        value = corr_matrix[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")
        
ax.set_xticks(np.arange(len(corr_vars)))
ax.set_yticks(np.arange(len(corr_vars)))

ax.set_xticklabels(corr_vars, rotation=45, ha="right")
ax.set_yticklabels(corr_vars)

plt.title("Correlation Matrix (Pearson)")
plt.tight_layout()

plt.show()