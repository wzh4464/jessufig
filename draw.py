import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def plot_line_chart(args):
    """
    Generates and saves a line chart based on the provided data and configuration parameters. 
    This function allows for customization of the chart's appearance and layout.

    Args:
        args (tuple): A tuple containing the following elements:
            - data (ndarray): A 2D array where the first row contains x-values and subsequent rows contain y-values.
            - pic_name (str): The name of the output picture file (without extension).
            - xlabel (str): The label for the x-axis.
            - ylabel (str): The label for the y-axis.
            - xlim (float): The upper limit for the x-axis.
            - ylim (float): The upper limit for the y-axis.
            - labellist (list): A list of labels for each line in the chart.
            - xticks (list): A list of ticks for the x-axis.
            - xlimD (float): The lower limit for the x-axis.
            - ylimD (float): The lower limit for the y-axis.
            - yticks (list): A list of ticks for the y-axis.
            - legend (bool): A flag indicating whether to display the legend.

    Returns:
        None: This function saves the plot to a file and does not return any value.

    Examples:
        plot_line_chart((data_array, 'my_chart', 'X Axis', 'Y Axis', 10, 100, 
                         ['Line 1', 'Line 2'], [0, 2, 4, 6, 8, 10], 
                         0, 10, [0, 20, 40, 60, 80, 100], True))
    """

    data, pic_name, xlabel, ylabel, xlim, ylim, labellist, xticks, xlimD, ylimD, yticks, legend = args
    
    x = data[0, :]
    ynum = np.size(data, 0) - 1
    y = data[1:ynum+1, :]
    
    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_alpha(0)  # Set figure background to transparent
    ax.patch.set_alpha(0)   # Set axes background to transparent
    
    markerlist = ['o', '^', 'x', 's', 'D']
    for i in range(ynum):
        ax.plot(x, y[i, :], label=labellist[i] if labellist else None, 
                marker=markerlist[i], markersize=6, linewidth=2)

    ax.grid(axis='both', linestyle='--', zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Adjust x-axis limits to prevent edge point clipping
    x_range = xlim - xlimD
    ax.set_xlim(xlimD - x_range/40, xlim + x_range/40)
    ax.set_xticks(xticks)
    
    ax.set_ylim(ylimD, ylim)
    ax.set_yticks(yticks)
    
    ax.tick_params(axis='both', labelsize=18)
    
    if legend:
        plt.rc('legend', fontsize=12)
        ax.legend(bbox_to_anchor=(1.0, 1.2), ncol=4, fancybox=True, shadow=True)

    plt.savefig(f"pic/{pic_name}.png", dpi=600, bbox_inches='tight', transparent=True)
    plt.close(fig)

def fMeasureArray(y_pre, y_recall):
    return 2 * y_pre * y_recall / (y_pre + y_recall)

def generate_plots_data(x, y_pre, y_recall, param_name, xlim, ylim, xticks, xlimD, ylimD, yticks):
    """
    Generates data for precision, recall, and F1 measure plots based on the provided input values. 
    This function structures the data and parameters needed for plotting these metrics.

    Args:
        x (ndarray): The x-values for the plots.
        y_pre (ndarray): The y-values representing precision.
        y_recall (ndarray): The y-values representing recall.
        param_name (str): The name of the parameter being plotted.
        xlim (float): The upper limit for the x-axis.
        ylim (float): The upper limit for the y-axis.
        xticks (list): A list of ticks for the x-axis.
        xlimD (float): The lower limit for the x-axis.
        ylimD (float): The lower limit for the y-axis.
        yticks (list): A list of ticks for the y-axis.

    Returns:
        list: A list of tuples, each containing data and parameters for precision, recall, and F1 measure plots.

    Examples:
        plots_data = generate_plots_data(x_values, precision_values, recall_values, 'beta', 10, 100, 
                                          [0, 2, 4, 6, 8, 10], 0, 10, [0, 20, 40, 60, 80, 100])
    """

    data_pre = np.vstack((x, y_pre))
    data_recall = np.vstack((x, y_recall))
    y_fMeasure = fMeasureArray(y_pre, y_recall)
    data_fMeasure = np.vstack((x, y_fMeasure))
    
    common_params = {
        'xlim': xlim,
        'ylim': ylim,
        'labellist': ['Ours', 'Jia', 'Shen', 'Meng', 'Zhao'],
        'xticks': xticks,
        'xlimD': xlimD,
        'ylimD': ylimD,
        'yticks': yticks
    }
    
    xlabel = r'$\beta$' if param_name == 'beta' else f'${param_name}$'
    
    return [
        (data_pre, f'Precision_{param_name}', xlabel, r"Precision (\%)", 
         xlim, ylim, common_params['labellist'], xticks, xlimD, ylimD, yticks, True),
        (data_recall, f'Recall_{param_name}', xlabel, r"Recall (\%)", 
         xlim, ylim, common_params['labellist'], xticks, xlimD, ylimD, yticks, False),
        (data_fMeasure, f'F1_{param_name}', xlabel, r"F Measure (\%)", 
         xlim, ylim, common_params['labellist'], xticks, xlimD, ylimD, yticks, False)
    ]

def main():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.size": 18,
        "mathtext.fontset": "stix",
    })

    # Beta plots
    x_beta = np.arange(6, 25, 3)
    y_pre_beta = np.array([
        [88.2, 80.2, 76.6, 72.1, 69.8, 66.0, 63.5],
        [63.6, 62.1, 63.0, 51.8, 42.4, 32.1, 29.8],
        [82.7, 78.2, 69.4, 61.9, 58.7, 54.7, 51.5],
        [84.2, 78.6, 72.1, 65.6, 62.3, 58.5, 55.1],
        [89.2, 82.2, 73.6, 66.1, 63.8, 59.0, 56.5]
    ])
    y_recall_beta = np.array([
        [86.7, 80.2, 71.6, 72.1, 69.8, 63.0, 58.5],
        [55.6, 41.8, 30.1, 25.4, 26.7, 21.0, 18.3],
        [62.2, 62.4, 53.1, 44.4, 45.7, 32.5, 26.5],
        [64.2, 58.6, 52.1, 45.6, 46.3, 41.5, 38.1], # should be higher
        [72.2, 62.2, 53.6, 46.1, 43.8, 38.0, 35.5] # should be higher
    ])
    beta_plots = generate_plots_data(x_beta, y_pre_beta, y_recall_beta, 'beta', 24, 100, x_beta, 6, 0, [0, 20, 40, 60, 80, 100])

    # d plots
    x_d = np.arange(0.06, 0.25, 0.03)
    y_pre_d = np.array([
        [81.7, 67.2, 52.6, 42.1, 21.6, 11.1, 5.6],
        [63.6, 20.6, 10.3, 5.1, 3.0, 1.5, 1.0],
        [92.2, 51.2, 25.6, 13.0, 6.5, 3.3, 1.7]
    ])
    y_recall_d = np.array([
        [76.7, 50.2, 21.6, 6.1, 2.8, 1.0, 0.5],
        [55.6, 11.8, 5.1, 2.4, 1.7, 1.0, 0.8],
        [62.2, 22.4, 13.1, 4.4, 2.7, 1.5, 1.3]
    ])
    d_plots = generate_plots_data(x_d, y_pre_d, y_recall_d, 'd', 0.24, 100, x_d, 0.06, 0, [0, 20, 40, 60, 80, 100])

    # IoU plots
    x_iou = np.arange(0.78, 0.97, 0.02)
    y_pre_iou = np.array([
        [81.7, 81.2, 80.6, 80.1, 79.6, 79.1, 78.6, 78.1, 75.6, 62.1],
        [63.6, 63.1, 62.6, 62.1, 61.6, 61.1, 57.9, 48.1, 35.6, 17.1],
        [88.5, 88.2, 88.1, 87.2, 87.0, 86.2, 82.8, 72.8, 62.2, 41.2]
    ])
    y_recall_iou = np.array([
        [76.7, 76.2, 75.6, 75.1, 74.6, 74.1, 73.6, 73.1, 65.6, 42.1],
        [55.6, 55.1, 54.6, 54.1, 53.6, 53.1, 49.9, 40.1, 27.6, 9.1],
        [62.2, 61.9, 61.8, 60.9, 60.7, 59.9, 56.5, 46.5, 35.9, 14.9]
    ])
    iou_plots = generate_plots_data(x_iou, y_pre_iou, y_recall_iou, 'IoU', 0.96, 100, x_iou, 0.78, 0, [0, 20, 40, 60, 80, 100])

    # Combine all plot data
    # all_plots = beta_plots + d_plots + iou_plots
    

    # Use multiprocessing to generate plots in parallel
    with Pool(processes=min(9, cpu_count())) as pool:
        # pool.map(plot_line_chart, all_plots)
        pool.map(plot_line_chart, beta_plots)

if __name__ == "__main__":
    main()
