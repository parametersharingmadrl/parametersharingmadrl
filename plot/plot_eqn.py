import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# log((1-v)*.05/(1-v - g/n))/log(1- k/(n-1) +
# (k*g*(n-1)/(n*(n-1)))/((k/(n-1)+1)(1-v) + v/(n-1))) for v=0.1, n=20, g=.01, k=0 to 0.00001

matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 8,
    #"legend.fontsize": 5,
    "text.usetex": True,
    "pgf.rcfonts": False
});

plt.figure(figsize=(3, 2))

NUM_AGENTS = 3

x = np.geomspace(0.00000005, 0.00001, 40)

def func(val):
    c = 0.1
    I = 0.01
    e = 0.001
    n = NUM_AGENTS

    #val *= n

    y = np.log(((1-c)*e)/(1-c-(I/n)))/np.log(1 - (val/(n-1)) + val*I/(n*((1+(val/(n-1))) * (1 - c) + (c/(n-1)))))

    return y


y_arr = np.array([func(i) for i in x])

plt.plot(x, y_arr)
plt.xlabel("$\mathcal{K}_{\star}$", labelpad=3)
plt.ylabel("Number of Steps", labelpad=3)
plt.tight_layout()
plt.title("Convergence Rates Over $\mathcal{K}_{\star}$")
plt.xticks(ticks=[0,.000003,.000006,.000009],#,.000008,.00001
labels=['0','$1\\cdot10^{-6}$','$2\\cdot10^{-6}$','$3\\cdot10^{-6}$'])#,'$8\\cdot10^{-6}$','$10\\cdot10^{-6}$'])

plt.yticks(ticks=[0,10e7,20e7,30e7],labels=['0','$1\\cdot10^{8}$','$2\\cdot10^{8}$','$3\\cdot10^{8}$'])

plt.savefig("kplot_camera.pgf", bbox_inches='tight', pad_inches=.025)
