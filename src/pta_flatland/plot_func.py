def set_custom_tick_options(ax, width=1, lenght=3):
    
    ax.minorticks_on()
    
    ax.tick_params(which='major', direction='in', 
                   length=2*lenght, width = width, 
                   bottom = True, 
                   top = True,
                   left = True,
                   right = True,
                   pad = 5)
    ax.tick_params(which='minor',direction='in',
                   length = lenght, width = width, 
                   bottom = True, 
                   top = True,
                   left = True,
                   right = True)

#==============================================================================

def set_size(width, fraction=1, ratio=None, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    fig_width_pt = width * fraction

    inches_per_pt = 1 / 72.27

    fig_width_in = fig_width_pt * inches_per_pt
    if not ratio:
        ratio = (5**0.5 - 1) / 2

    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def square_panels(fig, pad=0.06, wspace=None, n_iter=3):
    """Set margins from the rendered labels, then resize the figure so every
    axes is square.

    Call last, after every label / title / legend exists. The figure WIDTH is
    preserved -- it is what has to match the LaTeX text width -- and the
    height is solved for. Do not combine with tight_layout() or
    bbox_inches="tight": they fight over the same margins and undo the
    height fix.

    pad    : breathing space around the content, in INCHES (equal on all sides)
    wspace : gap between columns. None measures it from the inner y-labels.
             fig.get_tightbbox() only reports the outer extents of the whole
             grid, so an inner ylabel is invisible to it and a hardcoded
             wspace will let neighbouring panels overwrite each other.
    """
    W = fig.get_size_inches()[0]
    ncols = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[1]
    to_in = fig.dpi_scale_trans.inverted()

    for _ in range(n_iter):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        Wc, Hc = fig.get_size_inches()
        tb = fig.get_tightbbox(r)
        pos = [a.get_position() for a in fig.axes]

        # margins the content actually needs, in inches
        l_in = min(p.x0 for p in pos) * Wc - tb.x0 + pad
        r_in = tb.x1 - max(p.x1 for p in pos) * Wc + pad
        b_in = min(p.y0 for p in pos) * Hc - tb.y0 + pad
        t_in = tb.y1 - max(p.y1 for p in pos) * Hc + pad

        if wspace is None:
            # widest left-overhang among panels that are not in column 0
            over = [a.get_position().x0 * Wc
                    - a.get_tightbbox(r).transformed(to_in).x0
                    for a in fig.axes
                    if a.get_subplotspec().colspan.start > 0]
            gap_in = (max(over) if over else 0.0) + pad
        else:
            gap_in = None

        if gap_in is None:
            ax_w = W * (1 - (l_in + r_in) / W) / (ncols + (ncols - 1) * wspace)
            ws = wspace
        else:
            ax_w = (W - l_in - r_in - (ncols - 1) * gap_in) / ncols
            ws = gap_in / ax_w

        H = ax_w + b_in + t_in          # exact: makes axes height == ax_w
        fig.set_size_inches(W, H)
        fig.subplots_adjust(left=l_in / W, right=1 - r_in / W,
                            bottom=b_in / H, top=1 - t_in / H, wspace=ws)
    return fig
