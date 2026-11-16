"""Headless scientific plots with separate lines for supported segments."""
import os
from pathlib import Path
import numpy as np


def plot_series(path, series, title, xlabel, ylabel, *, connect=True, labels=None):
    """Export SVG and PNG without joining missing regions or overwriting files."""
    arrays=[]
    names=[]
    for i,s in enumerate(series):
        a=np.asarray(s,float).reshape(-1,2)
        a=a[np.isfinite(a).all(axis=1)]
        if len(a):
            arrays.append(a)
            names.append(None if labels is None else labels[i])
    if not arrays:
        return False
    path=Path(path)
    png=path.with_suffix('.png')
    if path.exists() or png.exists():
        raise FileExistsError(path)
    cache=path.parent/'.matplotlib'
    cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise RuntimeError('Install analysis plotting dependency: python -m pip install matplotlib') from exc
    fig=Figure(figsize=(9,4.8),layout='constrained')
    FigureCanvasAgg(fig)
    ax=fig.subplots()
    for array,name in zip(arrays,names):
        ax.plot(array[:,0],array[:,1],'.-' if connect else '.',linewidth=1,markersize=3,label=name)
    if labels is not None:
        ax.legend()
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    ax.grid(alpha=.25)
    ax.ticklabel_format(useOffset=False,style='plain')
    for destination,fmt in ((path,'svg'),(png,'png')):
        with destination.open('xb') as stream:
            fig.savefig(stream,format=fmt,dpi=150)
    fig.clear()
    return True
