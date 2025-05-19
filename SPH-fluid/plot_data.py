import json
import matplotlib.pyplot as plt

def load_pcg_iter_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        pcg_iter = data["residual_data"]["pcg_iter"]

        return pcg_iter

file1 = "./data/results/20250519_222827.json"
file2 = "./data/results/20250519_221831.json"

pcg_iter_1 = load_pcg_iter_from_json(file1)
pcg_iter_2 = load_pcg_iter_from_json(file2)

plt.figure(figsize=(8, 5))
plt.plot(pcg_iter_1, label='File 1', linestyle='-', linewidth=1)
plt.plot(pcg_iter_2, label='File 2', linestyle='--', linewidth=1)

plt.xlabel("Frame")
plt.ylabel("PCG Iteration Count")
plt.title("Comparison of PCG Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()