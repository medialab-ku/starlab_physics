import json
import os
import matplotlib.pyplot as plt

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
    "./data/results/20250520_144315.json",
    "./data/results/20250520_150616.json",
    "./data/results/20250520_152134.json",
    "./data/results/20250520_153718.json",
    "./data/results/20250520_155657.json",
]

bar_labels = [os.path.basename(p).split(".")[0] for p in file_paths]

N_SAMPLES = 5000


def load_pcg_iter(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    pcg_iter = data["residual_data"]["opt_iter"]
    return pcg_iter


def truncated_mean(seq, limit):
    if not seq:
        return 0.0
    k = min(len(seq), limit)
    return sum(seq[:k]) / k
    # return max(seq[:k])


means = [
    truncated_mean(load_pcg_iter(path), N_SAMPLES)
    for path in file_paths
]

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
    }
)

plt.figure(figsize=(6, 4))


x0 = load_pcg_iter(file_paths[0])[16::16]
x2 = load_pcg_iter(file_paths[2])[16::16]
x4 = load_pcg_iter(file_paths[4])[16::16]

plt.plot(x0, label="$\\eta=0.9$", linestyle='-')
plt.plot(x2, label="$\\eta=0.7$", linestyle='-')
plt.plot(x4, label="$\\eta=0.5$", linestyle='-')
plt.legend()
# plt.plot(x_2, label="w/ filtering", linestyle='-', linewidth=1)


# x_pos = range(len(bar_labels))
# plt.bar(x_pos, means, width=0.6, edgecolor="black")
#
# plt.xticks(x_pos, bar_labels, rotation=15)
plt.xlabel("Frame", fontsize=15)
plt.ylabel(r"Solver iteration", fontsize=15)
# plt.title(r"PCG iteration comparison", fontsize=14, pad=10)
#
# plt.tight_layout()
#
# os.makedirs("./figures", exist_ok=True)
# out_path = "./pcg_iter_avg.pdf"
plt.savefig("eta_solve_iter.pdf", format="pdf")
# print(f"그래프 저장 완료: {out_path}")

plt.show()