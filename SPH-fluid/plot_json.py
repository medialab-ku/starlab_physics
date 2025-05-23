import matplotlib.pyplot as plt
import numpy as np
import json
import os

class JsonPlot:
    def __init__(self, filename, params, residual_data, elapsed_time, avg_density):
        self.filename = filename
        self.params = params
        self.residual_data = residual_data
        self.elapsed_time = elapsed_time
        self.avg_density = avg_density
        self.output_dir = os.path.join(os.getcwd(), "data", "results")
        os.makedirs(self.output_dir, exist_ok=True)

    def export_json(self):
        json_data = {
            "params": self.params,
            "residual_data": self.residual_data,
            "elapsed_data": self.elapsed_time,
            "avg_density": self.avg_density
        }
        json_path = os.path.join(self.output_dir, self.filename + ".json")

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)

    def plot_data(self):
        fig, ax1 = plt.subplots(figsize=(8, 4))

        color1 = 'tab:blue'
        ax1.set_xlabel("Frame")
        # ax1.set_ylabel("# Iteration")
        # ax1.plot(self.residual_data["pcg_iter"], color="blue", label="PCG Iteration")
        # ax1.plot(self.residual_data["opt_iter"], color="green", label="Optimization Iteration", linestyle="--")
        ax1.plot(self.avg_density["density"], color="green", label="avg density", linestyle="--")
        ax1.tick_params(axis='y')
        ax1.grid(True, alpha=0.3)

        # ax2 = ax1.twinx()  # 두 번째 y축 공유
        # color2 = 'tab:green'
        # ax2.set_ylabel("Elapsed Time (s)")
        # ax2.plot(self.elapsed_time["elapsed_time"], color="orange", label="Elapsed Time", linewidth=1, linestyle='--')
        # ax2.tick_params(axis='y')

        plt.title("PCG Iteration vs Elapsed Time per Frame")
        fig.tight_layout()

        plot_path = os.path.join(self.output_dir, self.filename + ".png")
        plt.savefig(plot_path)

        plt.show()

        # plt.figure(figsize=(7, 5))
        # plt.plot(self.residual_data["pcg_iter"], label="PCG Iteration", linestyle="-", linewidth=1)
        # plt.plot(self.residual_data["opt_iter"], label="Optimization Iteration", linestyle="--", linewidth=1)
        # plt.xlabel("Frame")
        # plt.ylabel("# Iteration")
        # plt.title("Iteration")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        #
        # plot_path = os.path.join(self.output_dir, self.filename + ".png")
        # plt.savefig(plot_path)
        #
        # plt.show()