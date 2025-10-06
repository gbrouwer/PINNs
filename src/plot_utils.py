
# plot_utils.py
# Simple Matplotlib helpers for the PINN examples.
# Rules: use matplotlib, one chart per figure (no subplots), no explicit color choices.

import matplotlib.pyplot as plt
import numpy as np

def _save_or_show(fname=None, dpi=140):
    if fname:
        plt.tight_layout()
        plt.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_timeseries(t, series, labels=None, title=None, fname=None):
    plt.figure()
    t = np.asarray(t).reshape(-1)
    if isinstance(series, (list, tuple)):
        for s in series:
            plt.plot(t, np.asarray(s).reshape(-1))
        if labels:
            plt.legend(labels)
    else:
        plt.plot(t, np.asarray(series).reshape(-1))
    if title:
        plt.title(title)
    plt.xlabel("t")
    _save_or_show(fname)

def plot_scatter_xy(x, y, title=None, fname=None):
    plt.figure()
    plt.plot(np.asarray(x).reshape(-1), np.asarray(y).reshape(-1))
    if title:
        plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    _save_or_show(fname)

def plot_heatmap_grid(X, Y, Z, title=None, fname=None, extent=None, origin='lower', aspect='auto'):
    plt.figure()
    if X is not None and Y is not None:
        # assume regular grid
        im = plt.imshow(Z, origin=origin, aspect=aspect,
                        extent=extent)
    else:
        im = plt.imshow(Z, origin=origin, aspect=aspect)
    plt.colorbar(im)
    if title:
        plt.title(title)
    _save_or_show(fname)

def plot_quiver(X, Y, U, V, title=None, fname=None, step=4):
    plt.figure()
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Us = U[::step, ::step]
    Vs = V[::step, ::step]
    plt.quiver(Xs, Ys, Us, Vs)
    if title:
        plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    _save_or_show(fname)

def plot_hist(values, bins=30, title=None, fname=None):
    plt.figure()
    plt.hist(np.asarray(values).reshape(-1), bins=bins)
    if title:
        plt.title(title)
    _save_or_show(fname)

def plot_param_convergence(history_dict, title=None, fname=None):
    # history_dict: {"name": [values over iters], ...}
    plt.figure()
    for k, v in history_dict.items():
        plt.plot(np.asarray(v).reshape(-1), label=k)
    plt.legend()
    if title:
        plt.title(title)
    plt.xlabel("iteration")
    _save_or_show(fname)
