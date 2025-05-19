import json
import matplotlib.pyplot as plt

def load_pcg_iter_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        pcg_iter = data["residual_data"]["opt_iter"]

        return pcg_iter

def load_error_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        error = data["error"]

        return error


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": r"""
        \usepackage{newtxtext,newtxmath}
    """,
})

file1 = "./data/results/1.01.json"
file2 = "./data/results/1.001.json"

# x_1 = load_pcg_iter_from_json(file1)
# x_2 = load_pcg_iter_from_json(file2)

x_1 = load_error_from_json("./data/error/error0.json")
x_2 = load_error_from_json("./data/error/error.json")

plt.figure(figsize=(8, 3))
plt.plot(x_1, label="w/o filtering", linestyle='-', linewidth=1)
plt.plot(x_2, label="w/ filtering", linestyle='-', linewidth=1)

plt.xlabel("Solver iteration", fontsize=15,  labelpad=15)
plt.xlim(0, 160)
plt.ylabel("$\| \\Delta \mathbf{x}\|_{\\infty}$",  labelpad=10,  fontsize=15)
# plt.title("Comparison of PCG Iterations")
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig("div_error.pdf", format='pdf')
plt.show()