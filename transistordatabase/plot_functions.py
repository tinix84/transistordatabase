"""Contains functions for plotting that are used in the transistordatabase package."""
import matplotlib.pyplot as plt
from transistordatabase.helper_functions import get_img_raw_data

def plot_soa_lib(soa, buffer_req: bool = False):
    """
    Plot and convert safe operating region characteristic plots in raw data format.

    :param soa: List of SOA curves
    :param buffer_req: Internally required for generating virtual datasheets

    :return: Respective plots are displayed or raw data is returned
    """
    if not soa:
        return None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if isinstance(soa, list) and soa:
        for curve in soa:
            line1, = curve.get_plots(ax)
    plt.xlabel('$V_{ds}$ / $V_r$ [V]')
    plt.ylabel('$I_d$ / $I_r$ [A]')
    props = dict(fill=False, edgecolor='black', linewidth=1)
    if len(soa):
        plt.legend(fontsize=8)
        r_on_condition = '\n'.join(["conditions: ", "$T_{c} $ =" + str(soa[0].t_c) + " [°C]"])
        ax.text(0.65, 0.1, r_on_condition, transform=ax.transAxes, fontsize='small', bbox=props, ha='left', va='bottom')
    plt.grid()
    if buffer_req:
        return get_img_raw_data(plt)
    else:
        plt.show()
        
    return None
