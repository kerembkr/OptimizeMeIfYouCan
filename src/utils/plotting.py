import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_contour(func, x_hist):

    # parameter space
    x = np.linspace(-4.5, 4.5, 100)
    y = np.linspace(-4.5, 4.5, 100)

    # function values
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
      for j in range(len(y)):
        Z[i, j], _ = func([x[i], y[j]])

    # optimization path
    theta_hist_x, theta_hist_y = zip(*x_hist[0])
    theta_hist_x_m, theta_hist_y_m = zip(*x_hist[1])
    theta_hist_x_nm, theta_hist_y_nm = zip(*x_hist[2])
    theta_hist_x_ada, theta_hist_y_ada = zip(*x_hist[3])
    theta_hist_x_rms, theta_hist_y_rms = zip(*x_hist[4])
    theta_hist_x_adam, theta_hist_y_adam = zip(*x_hist[5])

    # contour plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.contour(X, Y, Z.T, levels=np.logspace(-0.5, 5.0, 35), norm=LogNorm(), cmap=plt.cm.jet, alpha=0.3)
    ax.plot(theta_hist_x, theta_hist_y, label="GD", linewidth=2)
    ax.plot(theta_hist_x_m, theta_hist_y_m, label="Momentum", linewidth=2)
    ax.plot(theta_hist_x_nm, theta_hist_y_nm, label="Nesterov Momentum", linewidth=2)
    ax.plot(theta_hist_x_ada, theta_hist_y_ada, label="Adagrad", linewidth=2)
    ax.plot(theta_hist_x_rms, theta_hist_y_rms, label="RMSProp", linewidth=2)
    ax.plot(theta_hist_x_adam, theta_hist_y_adam, label="Adam", linewidth=2)

    ax.spines['top'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)

    ax.scatter(3, 0.5, c="k", marker="*", s=300)
    ax.legend(framealpha=1)
    plt.show()