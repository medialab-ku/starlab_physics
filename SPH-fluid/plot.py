import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Check if Times New Roman is available
font_path = None
for f in fm.findSystemFonts(fontext='ttf'):
    if "Times New Roman" in fm.FontProperties(fname=f).get_name():
        font_path = f
        break

if font_path is None:
    raise RuntimeError("Times New Roman font file not found.")

x0 = 1.0
x1 = 0.3

def f(x):
    return np.maximum(x - x0, 0)

def g(x, x0, eta):
    x_start = x1 + eta * (x0 - x1)
    t = x0 - x_start
    shift = t * pow(1, 3) - 0.5 * t * pow(1, 4)
    return np.piecewise(
        x,
        [x < x_start, (x >= x_start) & (x < x0), x >= x0],
        [lambda x: 0, lambda x: t * pow((x - x_start)/t, 3) - 0.5 * t * pow((x - x_start)/t, 4), lambda x: x - x0 + shift]
    )

def g_grad(x, x0, eta):
    x_start = x1 + eta * (x0 - x1)
    t = x0 - x_start
    return np.piecewise(
        x,
        [x < x_start, (x >= x_start) & (x < x0), x >= x0],
        [lambda x: 0, lambda x: 3 * pow((x - x_start)/t, 2) - 2 * pow((x - x_start)/t, 3), lambda x: 1.0]
    )

def g_hess(x, x0, eta):
    x_start = x1 + eta * (x0 - x1)
    t = x0 - x_start
    return np.piecewise(
        x,
        [x < x_start, (x >= x_start) & (x < x0), x >= x0],
        [lambda x: 0, lambda x: 6/t * ((x - x_start)/t - pow((x - x_start)/t, 2)), lambda x: 0.0]
    )

# Plot setup
x = np.linspace(0.1, 1.2, 200)

phi = (1 + math.sqrt(5)) / 2
width = 6
height = width / phi

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": r"""
        \usepackage{newtxtext,newtxmath}
    """,
})
plt.figure(figsize=(width, height))

# y_0 = g_grad(x, x0=1.0, eta=0.0)
# y_1 = g_grad(x, x0=1.0, eta=0.5)
# y_2 = g_grad(x, x0=1.0, eta=0.9)
#
# plt.plot(x, y_0, label=r"$\eta = 0.0$")
# plt.plot(x, y_1, label=r"$\eta = 0.5$")
# plt.plot(x, y_2, label=r"$\eta = 0.9$")
# plt.axvline(x0, color="gray", linestyle="--")
#
# plt.xlabel(r"$\rho$", labelpad=10, fontsize=20)
# plt.ylabel(r"$\frac{\partial f}{\partial \rho}$", rotation=0, labelpad=20, fontsize=20)

y_0 = g_hess(x, x0=1.0, eta=0.0)
y_1 = g_hess(x, x0=1.0, eta=0.5)
y_2 = g_hess(x, x0=1.0, eta=0.9)

plt.plot(x, y_0, label=r"$\eta = 0.0$")
plt.plot(x, y_1, label=r"$\eta = 0.5$")
plt.plot(x, y_2, label=r"$\eta = 0.9$")
plt.axvline(x0, color="gray", linestyle="--")

plt.xlabel(r"$\rho$", labelpad=5, fontsize=20)
plt.ylabel(r"$\frac{\partial^2 f}{\partial \rho^2}$", rotation=0, labelpad=20, fontsize=20)

# y_0 = g(x, x0=1.0, eta=0.0)
# y_1 = g(x, x0=1.0, eta=0.5)
# y_2 = g(x, x0=1.0, eta=0.9)
#
# plt.plot(x, y_0, label=r"$\eta = 0.0$")
# plt.plot(x, y_1, label=r"$\eta = 0.5$")
# plt.plot(x, y_2, label=r"$\eta = 0.9$")
# plt.axvline(x0, color="gray", linestyle="--")
#
# plt.xlabel(r"$\rho$", labelpad=5, fontsize=20)
# plt.ylabel(r"$f$", rotation=0, labelpad=20, fontsize=20)



plt.legend(loc="upper left")
plt.tight_layout()

plt.savefig("f_hess.pdf", format='pdf')
plt.show()