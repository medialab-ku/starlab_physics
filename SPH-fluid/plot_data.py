import json
import os
import matplotlib.pyplot as plt
import numpy as np

# def load_pcg_iter_from_json(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#         pcg_iter = data["residual_data"]["opt_iter"]
#
#         return pcg_iter
#
# def load_error_from_json(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#         error = data["error"]
#
#         return error
#
#
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "text.latex.preamble": r"""
#         \usepackage{newtxtext,newtxmath}
#     """,
# })
#
# file1 = "./data/results/1.01.json"
# file2 = "./data/results/1.001.json"
#
# # x_1 = load_pcg_iter_from_json(file1)
# # x_2 = load_pcg_iter_from_json(file2)
#
# x_1 = load_error_from_json("./data/error/error0.json")
# x_2 = load_error_from_json("./data/error/error.json")
#
# plt.figure(figsize=(8, 3))
# plt.plot(x_1, label="w/o filtering", linestyle='-', linewidth=1)
# plt.plot(x_2, label="w/ filtering", linestyle='-', linewidth=1)
#
# plt.xlabel("Solver iteration", fontsize=15,  labelpad=15)
# plt.xlim(0, 160)
# plt.ylabel("$\| \\Delta \mathbf{x}\|_{\\infty}$",  labelpad=10,  fontsize=15)
# # plt.title("Comparison of PCG Iterations")
# plt.legend()
# # plt.grid(True)
# plt.tight_layout()
# plt.savefig("div_error.pdf", format='pdf')
# plt.show()

file_paths = [
    # "./data/results/20250520_144315.json",
    # "./data/results/20250520_150616.json",
    # "./data/results/20250520_152134.json",
    # "./data/results/20250520_153718.json",
    # "./data/results/20250520_155657.json",
    "./data/results/20250524_141522.json"
]

density_file_paths = [

        # "./data/results/20250523_053508.json",
        # "./data/results/20250523_051402.json",
        # "./data/results/20250523_051957.json",
    "./data/results/20250524_141522.json"
]
bar_labels = [os.path.basename(p).split(".")[0] for p in file_paths]


def load_pcg_iter(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    pcg_iter = data["residual_data"]["pcg_iter"]
    return pcg_iter

def load_opt_iter(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    opt_iter = data["residual_data"]["opt_iter"]
    return opt_iter

def load_elapesd_time(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    elapesd_time = data["elapsed_data"]["elapsed_time"]
    return elapesd_time

def load_density(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    density = data["avg_density"]["density"]
    return density


def truncated_mean(seq, limit):
    if not seq:
        return 0.0
    k = min(len(seq), limit)
    return sum(seq[:k]) / k
    # return max(seq[:k])

opt_iter = load_opt_iter(file_paths[0])
N_SAMPLES = len(opt_iter)

means = truncated_mean(opt_iter, N_SAMPLES)
min_value = min(opt_iter)
max_value = max(opt_iter)
print("Avg. Optimization iteration:", means, "/", "Min iteration:", min_value, "/", "Max iteration:", max_value)

elapsed_time = load_elapesd_time(file_paths[0])
N_SAMPLES = len(elapsed_time)
means = truncated_mean(elapsed_time, N_SAMPLES)
min_value = min(elapsed_time)
max_value = max(elapsed_time)
print("Avg. elapsed time:", means, "/", "Min elapsed time:", min_value, "/", "Max elapsed time:", max_value)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
    }
)

plt.figure(figsize=(6, 3))

#
x0 = np.array(load_pcg_iter(density_file_paths[0]))
y0 = np.array(load_opt_iter(density_file_paths[0]))

z = x0 / y0
# x2 = load_pcg_iter(file_paths[2])[16::16]
# x4 = load_pcg_iter(file_paths[4])[16::16]
#
plt.plot(z)
# plt.ylim(0, 50)
# plt.plot(x2, label="$\\eta=0.7$", linestyle='-')
# plt.plot(x4, label="$\\eta=0.5$", linestyle='-')
# plt.legend()
# plt.plot(x_2, label="w/ filtering", linestyle='-', linewidth=1)
plt.ylabel(r"Avg. Optimization iteration", fontsize=15, labelpad=10)
plt.xlabel("Frame", fontsize=15, labelpad=10)
# plt.legend()



# x_pos = range(len(bar_labels))
# plt.bar(x_pos, means, width=0.6, edgecolor="black")
#
# plt.xticks(x_pos, bar_labels, rotation=15)

# d0 = load_density(density_file_paths[0])
# d1 = load_density(density_file_paths[1])
# d2 = load_density(density_file_paths[2])
# plt.plot(d0, label="Ours, k=$1\\textrm{e}^4$", linestyle='-')
# plt.plot(d2, label="Xie et al., k=$1\\textrm{e}^4$", linestyle='-')
# plt.plot(d1, label="Xie et al,, k=$1\\textrm{e}^5$", linestyle='-')
# plt.xlabel("Frame", fontsize=15, labelpad=10)
# plt.ylabel(r"Avg. density", fontsize=15, labelpad=10)
# plt.legend()


# plt.title(r"PCG iteration comparison", fontsize=14, pad=10)
#
plt.tight_layout()
#
# os.makedirs("./figures", exist_ok=True)
# out_path = "./pcg_iter_avg.pdf"
# plt.savefig("avg_density.pdf", format="pdf")
# print(f"그래프 저장 완료: {out_path}")

plt.show()