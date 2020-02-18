import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def savefig(fig, path, save_pickle=False):
    """save matplotlib figure

    Args:
        fig (matplotlib.figure.Figure): figure object
        path (str): [description]
        save_pickle (bool, optional): Defaults to True. Whether to pickle the
          figure object as well.
    """

    fig.savefig(path, bbox_inches="tight")
    if save_pickle:
        import matplotlib
        import pickle

        # the `inline` of IPython will fail the pickle/unpickle; if so, switch
        # the backend temporarily
        if "inline" in matplotlib.get_backend():
            raise (
                "warning: the `inline` of IPython will fail the pickle/"
                "unpickle. Please use `matplotlib.use` to switch to other "
                "backend."
            )
        else:
            with open(path + ".pkl", "wb") as f:
                pickle.dump(fig, f)


def heatmap(
    A,
    B=None,
    labels=None,
    title="",
    table_like=False,
    color_bar=False,
    ax=None,
):
    """Draw heatmap along with dots diagram for visualizing a weight matrix."""

    if ax is None:
        ax = plt.gca()

    # square shaped
    ax.set_aspect("equal", "box")

    # turn off the frame
    ax.set_frame_on(False)

    # want a more natural, table-like display
    if table_like:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(A.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(A.shape[1]) + 0.5, minor=False)

    # turn off all ticks
    ax.xaxis.set_tick_params(top=False, bottom=False)
    ax.yaxis.set_tick_params(left=False, right=False)

    # add labels
    if labels is not None:
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)

    ax.set_title(title)

    # draw heatmap
    A_normed = (A - A.min()) / (A.max() - A.min())
    heatmap = ax.pcolor(A_normed, cmap=plt.cm.Greys)

    # add dots
    if B is not None:
        assert B.shape == A.shape
        for (y, x), w in np.ndenumerate(B):
            r = 0.35 * np.sqrt(w / B.max())
            circle = plt.Circle(
                (x + 0.5, y + 0.5), radius=r, color="darkgreen"
            )
            ax.add_artist(circle)

    # add colorbar
    if color_bar:
        ax.get_figure().colorbar(heatmap, ticks=[0, 1], orientation="vertical")


def plot_cross_validation(
    x,
    scores,
    show_error=True,
    allow_missing=False,
    xlabel="",
    ylabel="",
    title="",
    xscale="log",
    yscale=None,
    ax=None,
):
    """Plot cross-validation curve with respect to some parameters.
    Parameters
    ----------
    x : array-like, shape (n_params, )
        The values of parameter

    scores: array-like, shape (n_params, n_folds)
        Each row store the CV results for one parameter value. Note that it may
          contain np.nan

    """
    if ax is None:
        ax = plt.gca()

    # axes style
    ax.grid()

    # plot curve
    if allow_missing:
        y = np.nanmean(scores, axis=1)
        err = np.nanstd(scores, axis=1)
    else:
        idx = ~np.isnan(scores).any(axis=1)
        x = np.asarray(x)[idx]
        y = np.mean(scores[idx], axis=1)
        err = np.std(scores[idx], axis=1)

    # style
    fmt = "o-"

    # plot curve
    if show_error:
        ax.errorbar(x, y, err, fmt=fmt)
    else:
        ax.plot(x, y, fmt=fmt)

    # set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # set axis limit
    if xscale is None:
        ax.set_xlim(xmin=0, xmax=max(x))
    elif xscale == "log":
        ax.set_xlim(xmin=min(x) / 2, xmax=max(x) * 2)

    # set axis scale
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)


def ternaryplot(
    root_proba,
    colors=["red", "green", "blue"],
    markers=["s", "D", "o"],
    fontsize=20,
):

    assert (min(np.reshape(root_proba, np.size(root_proba))) >= 0) & (
        max(np.reshape(root_proba, np.size(root_proba))) <= 1
    )

    import ternary

    figure, tax = ternary.figure(scale=1)
    figure.set_size_inches(10, 10)
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="blue", multiple=0.05, linewidth=0.5)
    tax.bottom_axis_label("Normal", fontsize=fontsize, color="brown")
    tax.right_axis_label("early MCI", fontsize=fontsize, color="brown")
    tax.left_axis_label("clinical MCI", fontsize=fontsize, color="brown")

    # plot the prediction boundary
    p = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    p1 = (0, 0.5, 0.5)
    p2 = (0.5, 0, 0.5)
    p3 = (0.5, 0.5, 0)
    tax.line(p, p1, linestyle="--", color="brown", linewidth=3)
    tax.line(p, p2, linestyle="--", color="brown", linewidth=3)
    tax.line(p, p3, linestyle="--", color="brown", linewidth=3)

    # plot scatter plot of the points

    tax.scatter(
        root_proba, s=1, linewidth=3.5, marker=markers[0], color=colors[0]
    )

    tax.ticks(axis="lbr", multiple=0.1, linewidth=1)
    tax.clear_matplotlib_ticks()
    tax.show()

    return figure, tax


def barplot(
    x,
    hue,
    hue_labels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    ax=None,
    cmap_name="Accent",
):
    """Plot the proportion of hue for each value of x.

    Parameters
    ----------
    x : array-like
        Group id.
    hue : array-like
        Hue id
    """
    if ax is None:
        ax = plt.gca()

    n_group = len(set(x))
    n_hue = len(set(hue))
    hue2id = {h: i for i, h in enumerate(list(set(hue)))}

    data = np.zeros([n_group, n_hue])
    for i in range(len(x)):
        data[x[i], hue2id[hue[i]]] += 1

    data = data / data.sum(axis=1)[:, None]

    # set the style
    default_color_list = ["lightgreen", "dodgerblue", "orangered", "black"]
    if n_hue <= len(default_color_list):
        colors = default_color_list[:n_hue]
    else:
        colors = plt.cm.get_cmap(cmap_name)(np.linspace(0, 1, n_hue))
    width = 0.5 / n_hue

    left = np.arange(n_group)
    for i in range(n_hue):
        _ = ax.bar(left + width * i, data[:, i], width, color=colors[i])

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(left + n_hue * width / 2)
    ax.set_xticklabels(map(str, range(n_group)))

    # add ticks
    ax.yaxis.set_tick_params(left=True, right=True)
    ax.xaxis.set_tick_params(top=True, bottom=True)

    ax.set_xlim(xmin=-1 + width * n_hue, xmax=n_group)

    # add legend
    if hue_labels is not None and len(hue_labels) == n_hue:
        ax.legend(
            labels=hue_labels, loc="center left", bbox_to_anchor=(1, 0.5)
        )

    ax.set_title(title)


def networkplot(
    weights,
    labels=None,
    max_node_size=3000,
    min_node_size=100,
    max_width=10,
    min_width=1,
    arrowsize=80,
    colorbar=True,
    x_margins=0,
    scale=1.0,
    ax=None,
):
    import networkx as nx

    assert weights.shape[0] == weights.shape[1]
    assert labels is None or weights.shape[0] == len(labels)

    if not ax:
        ax = plt.figure().gca()

    G = nx.DiGraph()
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if weights[i][j] != 0:
                G.add_edge(i, j, weight=weights[i][j])

    pos = nx.spring_layout(G, seed=0)

    # node properties
    node_size = np.sqrt(np.abs(weights).sum(0))  # weight sum of outgoing edges

    node_size *= max_node_size / node_size.max()
    node_size = np.maximum(node_size, min_node_size)
    # node_size = max_node_size

    # edge properties
    edge_color = np.asarray([e[2]["weight"] for e in G.edges(data=True)])
    width = np.sqrt(np.abs(edge_color) / np.abs(edge_color).max()) * max_width
    width = np.maximum(width, min_width)

    if weights.min() >= 0:
        edge_vmax = weights.max()
        edge_vmin = 0
        cmap = plt.cm.Reds
    elif weights.max() <= 0:
        edge_vmax = 0
        edge_vmin = weights.min()
        cmap = plt.cm.Blues_r
    else:
        edge_vmax = np.abs(weights).max()
        edge_vmin = -edge_vmax
        # edge_vmax = weights.max()
        # edge_vmin = weights.min()
        cmap = plt.cm.RdYlBu_r

    nx.draw_networkx(
        G,
        pos,
        node_size=node_size,
        node_color="#008D0A",
        labels={k: labels[k] for k in pos},
        ax=ax,
    )

    # change arrow size
    arrowsize = width / width.max() * arrowsize

    for i, e in enumerate(G.edges()):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[e],
            arrowsize=arrowsize[i],
            width=width[i],
            edge_color=edge_color[i : i + 1],
            edge_cmap=cmap,
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
            node_size=node_size,
        )

    if colorbar:
        sm = mpl.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        )
        sm._A = []
        plt.colorbar(sm, ax=ax)

    ax.margins(x=x_margins)
    ax.axis("off")
    return ax
