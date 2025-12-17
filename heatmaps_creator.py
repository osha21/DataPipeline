import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.ndimage import gaussian_filter

X_MIN, X_MAX = -187.662, 62.853
Z_MIN, Z_MAX = -413.043, 80

NUM_BINS_X = 100
NUM_BINS_Z = 100

TELEMETRY_ROOT = "telemetry data"

DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop\heatmaps")
os.makedirs(DESKTOP_DIR, exist_ok=True)

folders = sorted([
    f for f in os.listdir(TELEMETRY_ROOT)
    if os.path.isdir(os.path.join(TELEMETRY_ROOT, f))
])

participant_index = 1

for folder_name in folders:
    folder_path = os.path.join(TELEMETRY_ROOT, folder_name)
    csv_path = os.path.join(folder_path, "positions.csv")

    xs = []
    zs = []

    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            xs.append(float(row["x"]))
            zs.append(float(row["z"]))

    xs = np.array(xs)
    zs = np.array(zs)

    heatmap, xedges, zedges = np.histogram2d(
        xs,
        zs,
        bins=[NUM_BINS_X, NUM_BINS_Z],
        range=[[X_MIN, X_MAX], [Z_MIN, Z_MAX]]
    )
    
    smoothed = gaussian_filter(heatmap, sigma=1.0)

    plt.figure(figsize=(10, 12))

    plt.imshow(
        smoothed.T,
        origin="lower",
        extent=[X_MIN, X_MAX, Z_MIN, Z_MAX],
        aspect="equal",
    )
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(f"Player Position Heatmap - Participant {participant_index}")
    plt.colorbar(label="Samples")
    plt.tight_layout()

    output_filename = f"heatmap_participant{participant_index}.png"
    output_path = os.path.join(DESKTOP_DIR, output_filename)

    plt.savefig(output_path, dpi=300)
    plt.close()

    participant_index += 1
