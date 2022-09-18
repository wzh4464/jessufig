# -*- coding: utf-8 -*-
# @Author: WU Zihan
# @Date:   2022-05-01 16:08:36
# @Last Modified by:   WU Zihan
# @Last Modified time: 2022-05-03 00:00:53
import gzip
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

FIG2_FULL = True


def analyze_a_log(file_path):
    node_id = ""
    start_time = 0
    receive_times = []
    with gzip.open(file_path, 'rt') as f:
        text = f.readlines()
        for i in text:
            if "NodeID is" in str(i):
                node_id = start_time = str(i).split(' ')[-1][:-1]
            if "Send msg at" in str(i):
                start_time = str(i).split(' ')[-1][:-1]

            if "Received a networkData originated from" in str(i):
                sender_id = str(i).split(' ')[-3]
                receive_time = str(i).split(' ')[-1][:-1]
                receive_times.append([sender_id, receive_time])
    return node_id, start_time, receive_times


def analyze_logs(dir_path):
    # Here dalays contains the delay of each msg this node received.
    node_start_times = {}
    all_receive_times = []
    g = os.walk(dir_path)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name == ".DS_Store":
                continue
            node_id, start_time, receive_times = analyze_a_log(dir_path + "/" + file_name)

            node_start_times[node_id] = start_time
            all_receive_times.append(receive_times)

    node_delays = {}
    for i in node_start_times.keys():
        node_delays[i] = []
    for receive_times in all_receive_times:
        for i in receive_times:
            delay = (float(i[1]) - float(node_start_times[i[0]])) / 1e9
            node_delays[i[0]].append(delay)

    for i in node_delays.keys():
        node_delays[i] = sorted(node_delays[i])

    counts = []
    full_count = 0
    for i in node_delays.keys():
        if len(node_delays[i]) != 0:
            counts.append(len(node_delays[i]))
        if len(node_delays[i]) == 99:
            full_count += 1
    print(full_count, "/", len(counts))
    print(min(counts))

    return node_delays


def analyze_multisource_delay(log_dir):
    all_node_delays = []
    g = os.walk(log_dir)
    for path, dir_list, file_list in g:
        for each_dir in sorted(dir_list):
            print(each_dir)
            if FIG2_FULL:
                node_delays = analyze_logs(log_dir + "/" + each_dir)
                all_node_delays.append(node_delays)
            else:

                if each_dir == "010s" or each_dir == "030s" or "old" in each_dir:
                    continue
                node_delays = analyze_logs(log_dir + "/" + each_dir)
                all_node_delays.append(node_delays)

    return all_node_delays


FIG2_FULL = True


def plot_fig1(dir_name):
    all_node_delays = analyze_multisource_delay(dir_name)
    source_delays = []
    for i in all_node_delays:
        each_source_delays = []
        for j in i.keys():
            if len(i[j]) != 0:
                each_source_delays.append(i[j])
        source_delays.append(np.mean(np.array(each_source_delays), axis=0))

    [m1, m2, m3, m4] = source_delays

    hf_index = int(len(m1) * 0.5)
    sf_index = int(len(m1) * 0.75)
    nn_index = int(len(m1) * 0.99)

    hf_delays = np.array([m1[hf_index], m2[hf_index], m3[hf_index], m4[hf_index]])
    sf_delays = np.array([m1[sf_index], m2[sf_index], m3[sf_index], m4[sf_index]])
    nn_delays = np.array([m1[nn_index], m2[nn_index], m3[nn_index], m4[nn_index]])

    msgsize_num = np.array([1, 2, 3, 4])

    plt.plot(msgsize_num, hf_delays, color='blue', label='50%', marker='o')
    plt.plot(msgsize_num, sf_delays, color='green', label='75%', marker='+')
    plt.plot(msgsize_num, nn_delays, color='red', label='99%', marker='x')
    plt.legend()
    plt.xlabel("Message size (MByte)")
    plt.ylabel("Network delay (s)")

    plt.savefig("Figure 1.png", dpi=600, bbox_inches='tight')
    plt.show()


def plot_fig2(dir_name):
    all_node_delays = analyze_multisource_delay(dir_name)
    source_delays = []
    for i in all_node_delays:
        each_source_delays = []
        for j in i.keys():
            if len(i[j]) != 0:
                each_source_delays.append(i[j])
        source_delays.append(np.mean(np.array(each_source_delays), axis=0))

    if not FIG2_FULL:
        [s1, s20, s40, s60, s80, s100] = source_delays
    else:
        [s1, s10, s20, s30, s40, s60, s80, s100] = source_delays

    hf_index = int(len(s1) * 0.5)
    sf_index = int(len(s1) * 0.75)
    nn_index = int(len(s1) * 0.99)

    if not FIG2_FULL:
        hf_delays = np.array([s1[hf_index], s20[hf_index], s40[hf_index], s60[hf_index], s80[hf_index], s100[hf_index]])
        sf_delays = np.array([s1[sf_index], s20[sf_index], s40[sf_index], s60[sf_index], s80[sf_index], s100[sf_index]])
        nn_delays = np.array([s1[nn_index], s20[nn_index], s40[nn_index], s60[nn_index], s80[nn_index], s100[nn_index]])
    else:
        hf_delays = np.array([s1[hf_index], s10[hf_index], s20[hf_index], s30[hf_index], s40[hf_index], s60[hf_index],
                              s80[hf_index], s100[hf_index]])
        sf_delays = np.array([s1[sf_index], s10[sf_index], s20[sf_index], s30[sf_index], s40[sf_index], s60[sf_index],
                              s80[sf_index], s100[sf_index]])
        nn_delays = np.array([s1[nn_index], s10[nn_index], s20[nn_index], s30[nn_index], s40[nn_index], s60[nn_index],
                              s80[nn_index], s100[nn_index]])

    if not FIG2_FULL:
        source_num = np.array([1, 20, 40, 60, 80, 100])
    else:
        source_num = np.array([1, 10, 20, 30, 40, 60, 80, 100])

    # plt.title("Figure 2")

    plt.plot(source_num, hf_delays, color='blue', label='50%', marker='o')
    plt.plot(source_num, sf_delays, color='green', label='75%', marker='+')
    plt.plot(source_num, nn_delays, color='red', label='99%', marker='x')
    plt.legend()
    plt.xlabel("Source number")
    plt.ylabel("Network delay (s)")

    plt.savefig("Figure 2.png", dpi=600, bbox_inches='tight')
    plt.show()


def plot_fig3(dir_name):
    all_node_delays = analyze_multisource_delay(dir_name)
    source_delays = []
    for i in all_node_delays:
        each_source_delays = []
        for j in i.keys():
            if len(i[j]) != 0:
                each_source_delays.append(i[j])
        source_delays.append(np.mean(np.array(each_source_delays), axis=0))
    s1 = source_delays[0]
    s1m2 = source_delays[1]
    s1m3 = source_delays[2]
    s1m4 = source_delays[3]
    s2m1 = source_delays[4]
    s3m1 = source_delays[5]
    s4m1 = source_delays[6]

    nn_index = int(len(s1) * 0.99)

    msg_nn_delays = np.array([s1[nn_index], s1m2[nn_index], s1m3[nn_index], s1m4[nn_index]])
    source_nn_delays = np.array([s1[nn_index], s2m1[nn_index], s3m1[nn_index], s4m1[nn_index]])

    x_pos = np.array([1, 2, 3, 4])



    plt.xticks(x_pos + 0.2, [1, 2, 3, 4])
    plt.bar(x_pos + 0.4, source_nn_delays, color='#e66101', label='99% with 1MByte msg', width=0.4)
    plt.bar(x_pos, msg_nn_delays, color='#fdb863', label='99% with 1 source', width=0.4)

    plt.legend()
    plt.xlabel("Amount of data (MByte)")
    plt.ylabel("Network delay (s)")

    plt.savefig("Figure 3.png", dpi=600, bbox_inches='tight')
    plt.show()


def plot_fig4(dir_name):
    tx_per_mb = 1820
    all_node_delays_1m = analyze_multisource_delay(dir_name + "/1m")
    source_delays_1m = []
    for i in all_node_delays_1m:
        each_source_delays = []
        for j in i.keys():
            if len(i[j]) != 0:
                each_source_delays.append(i[j])
        source_delays_1m.append(np.mean(np.array(each_source_delays), axis=0))

    all_node_delays_2m = analyze_multisource_delay(dir_name + "/2m")
    source_delays_2m = []
    for i in all_node_delays_2m:
        each_source_delays = []
        for j in i.keys():
            if len(i[j]) != 0:
                each_source_delays.append(i[j])
        source_delays_2m.append(np.mean(np.array(each_source_delays), axis=0))

    all_node_delays_3m = analyze_multisource_delay(dir_name + "/3m")
    source_delays_3m = []
    for i in all_node_delays_3m:
        each_source_delays = []
        for j in i.keys():
            if len(i[j]) != 0:
                each_source_delays.append(i[j])
        source_delays_3m.append(np.mean(np.array(each_source_delays), axis=0))

    all_node_delays_4m = analyze_multisource_delay(dir_name + "/4m")
    source_delays_4m = []
    for i in all_node_delays_4m:
        each_source_delays = []
        for j in i.keys():
            if len(i[j]) != 0:
                each_source_delays.append(i[j])
        source_delays_4m.append(np.mean(np.array(each_source_delays), axis=0))

    nn_index = int(len(all_node_delays_1m[0]) * 0.99) - 1

    throughput_s1 = 1 * np.array(
        [1 * tx_per_mb / source_delays_1m[0][nn_index], 2 * tx_per_mb / source_delays_2m[0][nn_index],
         3 * tx_per_mb / source_delays_3m[0][nn_index], 4 * tx_per_mb / source_delays_4m[0][nn_index]])
    throughput_s20 = 20 * np.array([1 * tx_per_mb / source_delays_1m[1][nn_index], 2 * tx_per_mb / source_delays_2m[1][
        nn_index], 3 * tx_per_mb / source_delays_3m[1][nn_index], 4 * tx_per_mb / source_delays_4m[1][nn_index]])
    throughput_s40 = 40 * np.array([1 * tx_per_mb / source_delays_1m[2][nn_index], 2 * tx_per_mb / source_delays_2m[2][
        nn_index], 3 * tx_per_mb / source_delays_3m[2][nn_index], 4 * tx_per_mb / source_delays_4m[2][nn_index]])
    throughput_s60 = 60 * np.array([1 * tx_per_mb / source_delays_1m[3][nn_index], 2 * tx_per_mb / source_delays_2m[3][
        nn_index], 3 * tx_per_mb / source_delays_3m[3][nn_index], 4 * tx_per_mb / source_delays_4m[3][nn_index]])
    throughput_s80 = 80 * np.array([1 * tx_per_mb / source_delays_1m[4][nn_index], 2 * tx_per_mb / source_delays_2m[4][
        nn_index], 3 * tx_per_mb / source_delays_3m[4][nn_index], 4 * tx_per_mb / source_delays_4m[4][nn_index]])
    throughput_s100 = 100 * np.array(
        [1 * tx_per_mb / source_delays_1m[5][nn_index], 2 * tx_per_mb / source_delays_2m[5][
            nn_index], 3 * tx_per_mb / source_delays_3m[5][nn_index], 4 * tx_per_mb / source_delays_4m[5][nn_index]])

    block_sizes = [1, 2, 3, 4]
    x = np.arange(4)
    plt.ylim((0, 7000))
    # plt.title("Figure 3")
    bar_width = 0.15
    plt.bar(x + 1 * bar_width, height=throughput_s1, width=bar_width, color='#ccebc5', label='1 source')
    plt.bar(x + 2 * bar_width, height=throughput_s20, width=bar_width, color='#a8ddb5', label='20 sources')
    plt.bar(x + 3 * bar_width, height=throughput_s40, width=bar_width, color='#7bccc4', label='40 sources')
    plt.bar(x + 4 * bar_width, height=throughput_s60, width=bar_width, color='#4eb3d3', label='60 sources')
    plt.bar(x + 5 * bar_width, height=throughput_s80, width=bar_width, color='#2b8cde', label='80 sources')
    # plt.bar(x + 6 * bar_width, height=throughput_s100, width=bar_width, color='#0868ac', label='100 sources')

    plt.xticks(x + 3 * bar_width, block_sizes)
    plt.legend(loc="upper left")
    plt.xlabel('Message size (MBytes)')
    plt.ylabel('Throughput (tps)')

    plt.savefig("Figure 4.png", dpi=600, bbox_inches='tight')
    plt.show()

def plot_fig5():
    x = [100, 150, 200, 250, 300]
    y2 = [43.2998006284934, 46.01244611735001, 49.15404783226302, 51.25115211007208, 52.80659091860833]
    y1 = [1647.7487326879796, 1541.9233353999998, 1444.5153721173185, 1407.7418970403025, 1367.527811170854]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    bar_width = 30
    ln2 = ax1.bar(x, y1, color='#2c7bb6', label='Throughput',width=bar_width,zorder=2)
    # pltx = plt.twinx()
    ln1 = ax2.plot(x, y2, color='#d73027', label='Latency', marker='^',zorder=3)
    # plt.plot(x, nn_delays, color='red', label='99%', marker='x')
    # plt.legend()
    ax1.set_xlabel("Committee size")
    ax1.set_ylabel("Throughput (tps)")
    ax2.set_ylabel("Latency (sec)")
    # lns = ln1 + ln2
    # labs = [l.get_label() for l in lns]
    ax1.set_ylim(1300, 1700)
    ax2.set_ylim(40,56)
    # print(ax1.viewLim)
    ax2.legend(loc=0)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y',linestyle='--',zorder=0)
    x_major_locator=MultipleLocator(50)
    ax1.xaxis.set_major_locator(x_major_locator)
    plt.savefig("Figure 5.png", dpi=600, bbox_inches='tight')
    plt.show()

def plot_fig6():
    x = [100, 150, 200, 250, 300]
    y2 = [37.500574698583115, 41.90798849859491, 43.28130229610142, 46.9003559087, 47.19135834462524]
    y1 = [3891.1790276808506, 3474.327983256944, 3364.1067355882356, 3104.4995919559324, 3085.3683290370373]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    bar_width = 30
    ln2 = ax1.bar(x, y1, color='#2c7bb6', label='Throughput',width=bar_width,zorder=2)
    # pltx = plt.twinx()
    ln1 = ax2.plot(x, y2, color='#d73027', label='Latency (sec)', marker='^',zorder=3)
    # plt.plot(x, nn_delays, color='red', label='99%', marker='x')
    # plt.legend()
    ax1.set_xlabel("Number of participating nodes")
    ax1.set_ylabel("Throughput (tps)")
    ax2.set_ylabel("Latency (sec)")
    # lns = ln1 + ln2
    # labs = [l.get_label() for l in lns]
    ax1.set_ylim(0, 4500)
    ax2.set_ylim(32,50)
    # print(ax1.viewLim)
    ax2.legend(loc='upper right')
    ax1.legend(loc='upper left')
    ax1.grid(axis='y',linestyle='--',zorder=0)
    x_major_locator=MultipleLocator(50)
    ax1.xaxis.set_major_locator(x_major_locator)
    plt.savefig("figparticipatingnodes.png", dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_figure = 5
    if plot_figure == 0:
        analyze_logs("../testbed/nodelog")
    elif plot_figure == 1:
        plot_fig1("figure1")
    elif plot_figure == 2:
        plot_fig2("figure2")
    elif plot_figure == 3:
        plot_fig3("figure3")
    elif plot_figure == 4:
        plot_fig4("figure4")
    elif plot_figure == 5:
        plot_fig5()
    elif plot_figure == 6:
        plot_fig6()
