import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")  # 使用非交互式后端
font_path = "/backup/codes/actions-runner/_work/PhDThesis/PhDThesis/Times New Roman.ttf"
font_prop = font_manager.FontProperties(fname=font_path)


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

    (
        data,
        pic_name,
        xlabel,
        ylabel,
        xlim,
        ylim,
        labellist,
        xticks,
        xlimD,
        ylimD,
        yticks,
        legend,
    ) = args

    x = data[0, :]
    ynum = np.size(data, 0) - 1
    y = data[1 : ynum + 1, :]

    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_alpha(0)  # Set figure background to transparent
    ax.patch.set_alpha(0)  # Set axes background to transparent

    markerlist = ["o", "^", "x", "s", "D"]
    for i in range(ynum):
        ax.plot(
            x,
            y[i, :],
            label=labellist[i] if labellist else None,
            marker=markerlist[i],
            markersize=6,
            linewidth=2,
        )

    ax.grid(axis="both", linestyle="--", zorder=0)
    label_size = 32
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontproperties=font_prop, fontsize=label_size)

    # Adjust x-axis limits to prevent edge point clipping
    x_range = xlim - xlimD
    ax.set_xlim(xlimD - x_range / 40, xlim + x_range / 40)
    ax.set_xticks(xticks)

    ax.set_ylim(ylimD, ylim)
    ax.set_yticks(yticks)

    ax.tick_params(axis="both", labelsize=20)

    if legend:
        plt.rc("legend", fontsize=20)
        ax.legend(bbox_to_anchor=(1.0, 1.4), ncol=3, fancybox=True, shadow=True)

    plt.savefig(f"pic/{pic_name}.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def fMeasureArray(y_pre, y_recall):
    return 2 * y_pre * y_recall / (y_pre + y_recall)


def generate_plots_data(
    x, y_pre, y_recall, param_name, xlim, ylim, xticks, xlimD, ylimD, yticks, suffix=""
):
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
        suffix (str): A suffix to add to the plot name for distinction (e.g., "over", "occ").

    Returns:
        list: A list of tuples, each containing data and parameters for precision, recall, and F1 measure plots.
    """

    data_pre = np.vstack((x, y_pre))
    data_recall = np.vstack((x, y_recall))
    y_fMeasure = fMeasureArray(y_pre, y_recall)
    data_fMeasure = np.vstack((x, y_fMeasure))

    common_params = {
        "xlim": xlim,
        "ylim": ylim,
        "labellist": ["Ours", "Jia", "Shen", "Meng", "Zhao"],
        "xticks": xticks,
        "xlimD": xlimD,
        "ylimD": ylimD,
        "yticks": yticks,
    }

    xlabel = r"$\beta$" if param_name == "beta" else f"${param_name}$"
    name_suffix = f"_{suffix}" if suffix else ""

    return [
        (
            data_pre,
            f"Precision_{param_name}{name_suffix}",
            xlabel,
            r"Precision (\%)",
            xlim,
            ylim,
            common_params["labellist"],
            xticks,
            xlimD,
            ylimD,
            yticks,
            True,
        ),
        (
            data_recall,
            f"Recall_{param_name}{name_suffix}",
            xlabel,
            r"Recall (\%)",
            xlim,
            ylim,
            common_params["labellist"],
            xticks,
            xlimD,
            ylimD,
            yticks,
            False,
        ),
        (
            data_fMeasure,
            f"F1_{param_name}{name_suffix}",
            xlabel,
            r"F-measure (\%)",
            xlim,
            ylim,
            common_params["labellist"],
            xticks,
            xlimD,
            ylimD,
            yticks,
            False,
        ),
    ]


def load_image(filename):
    return np.array(Image.open(f"pic/{filename}.png"))


def combine_plots(plot_files):
    """
    Combines multiple plot files into a single 3x3 grid image using optimized methods.

    Args:
        plot_files (list): A list of filenames for the plot images to combine.
                           The order should be: [precision_plots, recall_plots, f1_plots].

    Returns:
        None: Saves the combined plot as a single image file.
    """
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    fig.patch.set_facecolor("white")  # Set white background for the entire figure

    # Use ThreadPoolExecutor to load images in parallel
    with ThreadPoolExecutor(max_workers=9) as executor:
        images = list(executor.map(load_image, plot_files))

    for i, img in enumerate(images):
        row = i // 3
        col = i % 3
        axs[row, col].imshow(img)
        axs[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("pic/combined_plots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    # 更新全局字体设置
    plt.rcParams.update(
        {
            "text.usetex": True,  # 启用 LaTeX 渲染
            "text.latex.preamble": r"\usepackage{times}",  # 使用 Times New Roman
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.sans-serif": ["Times New Roman"],
            "font.monospace": ["Times New Roman"],
            "mathtext.fontset": "cm",  # 数学字体使用 Computer Modern
            "font.size": 18,
        }
    )

    # Overlap plots
    x_beta = np.arange(6, 25, 3)
    y_pre_beta = np.array(
        [
            [88.2, 80.2, 76.6, 72.1, 69.8, 66.0, 63.5],
            [63.6, 62.1, 63.0, 51.8, 38.4, 32.1, 29.8],
            [82.7, 78.2, 69.4, 61.9, 52.7, 43.7, 36.5],
            [84.2, 78.6, 72.1, 65.6, 50.3, 45.5, 38.1],
            [89.2, 82.2, 73.6, 66.1, 60.8, 52.0, 44.5],
        ]
    )
    y_recall_beta = np.array(
        [
            [86.7, 80.2, 71.6, 72.1, 69.8, 63.0, 58.5],
            [55.6, 41.8, 30.1, 25.4, 26.7, 21.0, 18.3],
            [62.2, 62.4, 53.1, 44.4, 45.7, 32.5, 26.5],
            [64.2, 63.6, 60.1, 55.6, 56.3, 51.5, 48.1],
            [69.2, 65.2, 56.6, 49.1, 46.8, 41.0, 38.5],
        ]
    )
    over_plots = generate_plots_data(
        x_beta,
        y_pre_beta,
        y_recall_beta,
        "beta",
        24,
        100,
        x_beta,
        6,
        0,
        [0, 20, 40, 60, 80, 100],
        "over",
    )

    # Occluded plots
    x_beta = np.arange(6, 25, 3)
    y_pre_beta = np.array(
        [
            [86.2, 79.2, 77.6, 72.1, 64.8, 62.0, 59.5],
            [63.6, 60.1, 48.0, 37.8, 30.4, 22.1, 19.8],
            [80.7, 76.2, 67.4, 59.9, 57.7, 54.7, 51.5],
            [79.2, 73.6, 67.1, 60.6, 57.3, 53.5, 50.1],
            [84.2, 80.2, 73.6, 66.1, 63.8, 59.0, 56.5],
        ]
    )
    y_recall_beta = np.array(
        [
            [86.7, 70.2, 61.6, 62.1, 59.8, 53.0, 48.5],
            [55.6, 31.8, 20.1, 15.4, 10.7, 6.0, 3.3],
            [62.2, 52.4, 43.1, 29.4, 26.8, 11.0, 8.5],
            [64.2, 53.6, 40.1, 35.6, 36.3, 31.5, 28.1],
            [69.2, 55.2, 46.6, 34.1, 35.7, 22.5, 16.5],
        ]
    )
    occ_plots = generate_plots_data(
        x_beta,
        y_pre_beta,
        y_recall_beta,
        "beta",
        24,
        100,
        x_beta,
        6,
        0,
        [0, 20, 40, 60, 80, 100],
        "occ",
    )

    # d plots
    x_d = np.arange(0.06, 0.25, 0.03)
    y_pre_d = np.array(
        [
            [91.7, 71.2, 50.6, 25.1, 12.6, 6.1, 3.6],
            [63.6, 20.6, 10.3, 5.1, 3.0, 1.5, 1.0],
            [92.2, 51.2, 25.6, 13.0, 6.5, 3.3, 1.7],
            [90.2, 65.2, 35.6, 21.1, 10.6, 5.1, 2.6],
            [95.2, 75.2, 45.6, 11.1, 8.6, 4.1, 1.6],
        ]
    )
    y_recall_d = np.array(
        [
            [76.7, 65.2, 45.6, 25.1, 12.6, 6.1, 3.6],
            [55.6, 11.8, 5.1, 2.4, 1.7, 1.0, 0.8],
            [62.2, 22.4, 13.1, 4.4, 2.7, 1.5, 1.3],
            [64.2, 32.6, 15.1, 6.6, 3.3, 1.8, 1.3],
            [69.2, 45.2, 21.6, 6.1, 4.8, 2.1, 1.6],
        ]
    )
    d_plots = generate_plots_data(
        x_d, y_pre_d, y_recall_d, "d", 0.24, 100, x_d, 0.06, 0, [0, 20, 40, 60, 80, 100]
    )

    # Combine all plot data
    all_plots = over_plots + occ_plots + d_plots

    # Use multiprocessing to generate plots in parallel
    with Pool(processes=min(9, cpu_count())) as pool:
        pool.map(plot_line_chart, all_plots)

    plot_files = [
        "Precision_beta_over",
        "Precision_beta_occ",
        "Precision_d",
        "Recall_beta_over",
        "Recall_beta_occ",
        "Recall_d",
        "F1_beta_over",
        "F1_beta_occ",
        "F1_d",
    ]
    combine_plots(plot_files)


if __name__ == "__main__":
    main()
