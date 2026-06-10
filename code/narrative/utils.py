def plot_style():
    import seaborn as sns, matplotlib.pyplot as plt, matplotlib as mpl

    # sets fontsize etc. appropriate for presentation/paper, etc.
    sns.set(context='talk', style='white', palette='deep')

    # keep text editable in svg
    plt.rcParams['svg.fonttype'] = 'none'

    # push ticks inward
    mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
    # remove top and right splines
    mpl.rcParams['axes.spines.top'], mpl.rcParams['axes.spines.right'] = False, False