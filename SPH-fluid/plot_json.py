import matplotlib.pyplot as plt
import numpy as np
import json
import os

class JsonPlot:
    def __init__(self, filename, params, residual_data):
        self.filename = filename
        self.params = params
        self.residual_data = residual_data
        self.output_dir = os.path.join(os.getcwd(), "data", "results")
        os.makedirs(self.output_dir, exist_ok=True)

    def export_json(self):
        json_data = {
            "params": self.params,
            "residual_data": self.residual_data
        }
        json_path = os.path.join(self.output_dir, self.filename + ".json")

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)

    def plot_data(self):
        plt.figure(figsize=(7, 5))
        plt.plot(self.residual_data["pcg_iter"], label="PCG Iteration", linestyle="-", linewidth=1)
        plt.plot(self.residual_data["opt_iter"], label="Optimization Iteration", linestyle="--", linewidth=1)
        plt.xlabel("Frame")
        plt.ylabel("# Iteration")
        plt.title("Iteration")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, self.filename + ".png")
        plt.savefig(plot_path)

        plt.show()