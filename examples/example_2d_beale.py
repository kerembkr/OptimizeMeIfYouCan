import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from input.testfuncs_2d import beale
from src.utils.plotting import plot_contour


def Adam(func, theta, gamma=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8, epochs=100):

    theta_hist = [theta.copy()]

    # init momentum terms
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    # optimization loop
    for i in range(epochs):

        # Standard momentum
        f, df = func(theta)

        # 1st and 2nd momentum
        m = beta_1 * m + (1-beta_1) * df
        v = beta_2 * v + (1-beta_2) * df**2

        # bias correction
        m_hat = m/(1-beta_1)
        v_hat = v/(1-beta_2)

        # update parameters
        theta = theta - gamma * m_hat / (np.sqrt(v_hat) + eps)

        # Save parameter values
        theta_hist.append(theta.copy())

    return theta, theta_hist

def GD(func, theta, gamma=0.001, beta=0.0, epochs=100, nesterov=False):

    theta_hist = [theta.copy()]

    # initial momentum term is zero
    z = np.zeros_like(theta)

    # optimization loop
    for i in range(epochs):
        if nesterov:
            # Nesterov momentum: lookahead step
            f, df = func(theta - beta * z)
        else:
            # Standard momentum
            f, df = func(theta)

        # Update momentum term
        z = beta * z + gamma * df

        # Update parameters
        theta = theta - z

        # Save parameter values
        theta_hist.append(theta.copy())

    return theta, theta_hist

def Adagrad(func, theta, gamma=0.001, eps=1e-8, epochs=100):

    theta_hist = [theta.copy()]

    # initial momentum term
    G = np.zeros_like(theta)
    G.fill(0.2)

    # optimization loop
    for i in range(epochs):

        # Standard momentum
        f, df = func(theta)

        # past gradient values
        G = G + df**2

        # adaptive gradient
        df_adj = df / (np.sqrt(G) + eps)

        # update parameters
        #theta = theta - gamma * df_adj
        theta = theta - gamma * df / (np.sqrt(G) + eps)

        # Save parameter values
        theta_hist.append(theta.copy())

    return theta, theta_hist

def RMSProp(func, theta, gamma=0.001, alpha=0.99, eps=1e-8, epochs=100):

    theta_hist = [theta.copy()]

    # initial momentum term
    G = np.zeros_like(theta)
    G.fill(0.2)

    # optimization loop
    for i in range(epochs):

        # Standard momentum
        f, df = func(theta)

        # past gradient values
        # G = G + df**2
        G = alpha * G + (1-alpha) * df**2


        # adaptive gradient
        df_adj = df / (np.sqrt(G) + eps)

        # update parameters
        #theta = theta - gamma * df_adj
        theta = theta - gamma * df / (np.sqrt(G) + eps)

        # Save parameter values
        theta_hist.append(theta.copy())

    return theta, theta_hist

# choose objective function (only Beale function implemented, so I would recommend to choose Beale)
func = beale

epochs = 1000

# initial value
theta0 = np.array([0.7, 1.4])

# perform optimizations
_, loss_gd = GD(func, theta0, gamma=0.001, beta=0.0, epochs=epochs)  # without momentum
_, loss_gdm = GD(func, theta0, gamma=0.001, beta=0.95, epochs=epochs)  # with momentum
_, loss_gdnm = GD(func, theta0, gamma=0.001, beta=0.95, epochs=epochs, nesterov=True)  # with Nesterov momentum
_, loss_ada = Adagrad(func, theta0, gamma=0.2, epochs=epochs)  # Adaptive step-size
_, loss_rms = RMSProp(func, theta0, gamma=0.01, alpha=0.9, epochs=epochs)  # Adaptive step-size RMSProp
_, loss_adam = Adam(func, theta0, gamma=0.01, beta_1=0.9, beta_2=0.999, epochs=epochs)  # Adaptive step-size and momentum Adam
# plot optimization paths
plot_contour(func, [loss_gd, loss_gdm, loss_gdnm, loss_ada, loss_rms, loss_adam])