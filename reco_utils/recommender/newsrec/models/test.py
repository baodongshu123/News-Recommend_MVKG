import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker

def hua_improved_MVKG_dropout_AUC():
    # Data for AUC
    # Set global font size
    plt.rcParams.update({'font.size': 13})  # 设置全局字体大小为14
    auc_data = [(0.0, 0.6621), (0.1, 0.6668), (0.2, 0.6690), (0.3, 0.6645), (0.4, 0.6612), (0.5, 0.6557), (0.6, 0.6528)]

    # Extract x and y values from auc_data
    auc_x, auc_y = zip(*auc_data)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the line graph for AUC data
    ax.plot(auc_x, auc_y, marker='o', linestyle='-', color='b')

    # Hide the x-axis
    ax.xaxis.set_visible(False)

    # Set the y-axis limits and ticks
    ax.set_ylim(0.64, 0.68)
    ax.set_yticks([0.64, 0.65, 0.66, 0.67, 0.68])
    ax.set_yticklabels(['0.64', '0.65', '0.66', '0.67', '0.68'])

    # Set the aspect ratio to be equal, so 1 cm in x equals 1 cm in y
    ax.set_aspect(aspect=3.5)

    # Set the top and bottom lines to dashed with larger gaps

    ax.spines['bottom'].set_linestyle((0, (5, 10)))

    # Add labels and title
    ax.set_ylabel('AUC ')
    plt.savefig('improved_auc.png')

    # Show the plot
    plt.show()
def hua_improved_MVKG_dropout_MRR():
    # Data for MRR
    plt.rcParams.update({'font.size': 13})  # 设置全局字体大小为14
    mrr_data = [(0.0, 0.3046), (0.1, 0.308), (0.2, 0.3137), (0.3, 0.3087), (0.4, 0.3055), (0.5, 0.3039), (0.6, 0.3027)]

    # Extract x and y values from mrr_data
    mrr_x, mrr_y = zip(*mrr_data)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the line graph for MRR data
    ax.plot(mrr_x, mrr_y, marker='o', linestyle='-', color='g')

    # Set the x-axis label and ticks
    ax.set_xlabel('Mask probability p')

    ax.set_xticklabels(['0', '0.0', '0.1', '0.2', '0.3', '0.4', '0.5','0.6'])

    # Set the y-axis limits and ticks
    ax.set_ylim(0.30, 0.32)
    ax.set_yticks([0.30, 0.31, 0.32])
    ax.set_yticklabels(['0.30', '0.31', '0.32'])

    # Set the aspect ratio to be equal, so 1 cm in x equals 1 cm in y
    ax.set_aspect(aspect=3.5)

    # Set the top and bottom lines to dashed with larger gaps
    ax.spines['top'].set_linestyle((0, (5, 10)))


    # Add labels and title
    ax.set_ylabel('MRR')
    plt.savefig('improved_MRR.png')

    # Show the plot
    plt.show()
def hua_improved_MVKG_dropout_NDCG5():
    # Data for NDCG@5
    plt.rcParams.update({'font.size': 13})  # 设置全局字体大小为14
    ndcg5_data = [(0.0, 0.3418), (0.1, 0.3465), (0.2, 0.3515), (0.3, 0.3483), (0.4, 0.3432), (0.5, 0.3411),
                  (0.6, 0.3385)]

    # Extract x and y values from ndcg5_data
    ndcg5_x, ndcg5_y = zip(*ndcg5_data)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the line graph for NDCG@5 data
    ax.plot(ndcg5_x, ndcg5_y, marker='o', linestyle='-', color='r')

    # Hide the x-axis
    ax.xaxis.set_visible(False)

    # Set the y-axis limits and ticks
    ax.set_ylim(0.33, 0.36)
    ax.set_yticks([0.33, 0.34, 0.35, 0.36])
    ax.set_yticklabels(['0.33', '0.34', '0.35', '0.36'])

    # Set the aspect ratio to be equal, so 1 cm in x equals 1 cm in y
    ax.set_aspect(aspect=3.5)

    # Set the top and bottom lines to dashed with larger gaps
    ax.spines['top'].set_linestyle((0, (5, 10)))
    ax.spines['bottom'].set_linestyle((0, (5, 10)))

    # Add labels and title
    ax.set_ylabel('NDCG@5')
    # Save the plot as a PNG image
    plt.savefig('improved_NDCG5.png')

    # Show the plot
    plt.show()
def hua_improved_MVKG_dropout_NDCG10():
    # Data for NDCG@10
    plt.rcParams.update({'font.size': 13})  # 设置全局字体大小为14
    ndcg10_data = [(0.0, 0.4018), (0.1, 0.4063), (0.2, 0.4108), (0.3, 0.4068), (0.4, 0.4031), (0.5, 0.4001),
                   (0.6, 0.3983)]

    # Extract x and y values from ndcg10_data
    ndcg10_x, ndcg10_y = zip(*ndcg10_data)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the line graph for NDCG@10 data
    ax.plot(ndcg10_x, ndcg10_y, marker='o', linestyle='-', color='#00CED1')

    # Hide the x-axis
    ax.xaxis.set_visible(False)

    # Set the y-axis limits and ticks
    ax.set_ylim(0.39, 0.42)
    ax.set_yticks([0.39, 0.40, 0.41, 0.42])
    ax.set_yticklabels(['0.39', '0.40', '0.41', '0.42'])

    # Set the aspect ratio to be equal, so 1 cm in x equals 1 cm in y
    ax.set_aspect(aspect=3.5)

    # Set the top and bottom lines to dashed with larger gaps
    ax.spines['top'].set_linestyle((0, (5, 10)))
    ax.spines['bottom'].set_linestyle((0, (5, 10)))

    # Add labels and title
    ax.set_ylabel('NDCG@10')

    plt.savefig('improved_NDCG10.png')
    # Show the plot
    plt.show()
def hua_LSMV_negative_AUC():

    # Data for AUC
    plt.rcParams.update({'font.size': 13})  # Set the global font size to 14
    auc_data = [(1, 0.6537), (2, 0.6566), (3, 0.6583), (4, 0.6637), (5, 0.6587), (6, 0.6577), (7, 0.6572), (8, 0.6569),
                (9, 0.6566), (10, 0.6524)]

    # Extract x and y values from auc_data
    auc_x, auc_y = zip(*auc_data)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the line graph for AUC data
    ax.plot(auc_x, auc_y, marker='o', linestyle='-', color='b')

    # Hide the x-axis
    ax.xaxis.set_visible(False)

    # Set the y-axis limits and ticks
    ax.set_ylim(0.65, 0.67)
    ax.set_yticks([0.65, 0.66, 0.67])
    ax.set_yticklabels(['0.65', '0.66', '0.67',])

    # Set the aspect ratio to be equal, so 1 cm in x equals 1 cm in y
    ax.set_aspect(aspect=100)

    # Set the top and bottom lines to dashed with larger gaps

    ax.spines['bottom'].set_linestyle((0, (5, 10)))

    # Add labels and title
    ax.set_ylabel('AUC ')
    plt.savefig('auc_negative.png')

    # Show the plot
    plt.show()
def hua_LSMV_negative_MRR():
    # Data for MRR
    plt.rcParams.update({'font.size': 13})  # 设置全局字体大小为14
    mrr_data = [(1,0.3039), (2,0.3059), (3,0.309), (4,0.3106), (5,0.3077), (6,0.3069), (7,0.3058), (8,0.3067), (9,0.3067), (10,0.3031)]

    # Extract x and y values from mrr_data
    mrr_x, mrr_y = zip(*mrr_data)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the line graph for MRR data
    ax.plot(mrr_x, mrr_y, marker='o', linestyle='-', color='g')

    # Set the x-axis label and ticks
    ax.set_xlabel('Mask probability p')
    # Set the x-axis limits and ticks

    ax.set_xticklabels(['1','2', '4', '6', '8', '10', '8',])

    # Set the y-axis limits and ticks
    ax.set_ylim(0.3, 0.32)
    ax.set_yticks([0.30, 0.31, 0.32])
    ax.set_yticklabels(['0.30', '0.31', '0.32'])

    # Set the aspect ratio to be equal, so 1 cm in x equals 1 cm in y
    ax.set_aspect(aspect=90)

    # Set the top and bottom lines to dashed with larger gaps
    ax.spines['top'].set_linestyle((0, (5, 10)))


    # Add labels and title
    ax.set_ylabel('MRR')
    plt.savefig('MRR_negative.png')

    # Show the plot
    plt.show()
def hua_LSMV_negative_NDCG5():
    # Data for NDCG@5
    plt.rcParams.update({'font.size': 13})  # 设置全局字体大小为14

    ndcg5_data = [(1,0.3392), (2,0.3399), (3,0.3418), (4,0.3443), (5,0.3389), (6,0.3385), (7,0.3383), (8,0.3398), (9,0.338), (10,0.3325)]
    # Extract x and y values from ndcg5_data
    ndcg5_x, ndcg5_y = zip(*ndcg5_data)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the line graph for NDCG@5 data
    ax.plot(ndcg5_x, ndcg5_y, marker='o', linestyle='-', color='r')

    # Hide the x-axis
    ax.xaxis.set_visible(False)

    # Set the y-axis limits and ticks
    ax.set_ylim(0.33, 0.35)
    ax.set_yticks([0.33, 0.34, 0.35,])
    ax.set_yticklabels(['0.33', '0.34', '0.35',])

    # Set the aspect ratio to be equal, so 1 cm in x equals 1 cm in y
    ax.set_aspect(aspect=90)

    # Set the top and bottom lines to dashed with larger gaps
    ax.spines['top'].set_linestyle((0, (5, 10)))
    ax.spines['bottom'].set_linestyle((0, (5, 10)))

    # Add labels and title
    ax.set_ylabel('NDCG@5')
    # Save the plot as a PNG image
    plt.savefig('NDCG5_negative.png')

    # Show the plot
    plt.show()
def hua_LSMV_negative_NDCG10():
    # Data for NDCG@10
    plt.rcParams.update({'font.size': 13})  # 设置全局字体大小为14

    ndcg10_data = [(1,0.3987), (2,0.4007), (3,0.4019), (4,0.4058), (5,0.401), (6,0.40), (7,0.4016), (8,0.401), (9,0.3997),(10,0.3959)]
    # Extract x and y values from ndcg10_data
    ndcg10_x, ndcg10_y = zip(*ndcg10_data)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the line graph for NDCG@10 data
    ax.plot(ndcg10_x, ndcg10_y, marker='o', linestyle='-', color='#00CED1')

    # Hide the x-axis
    ax.xaxis.set_visible(False)

    # Set the y-axis limits and ticks
    ax.set_ylim(0.39, 0.41)
    ax.set_yticks([0.39, 0.40, 0.41,])
    ax.set_yticklabels([ '0.39','0.40', '0.41'])

    # Set the aspect ratio to be equal, so 1 cm in x equals 1 cm in y
    ax.set_aspect(aspect=90)

    # Set the top and bottom lines to dashed with larger gaps
    ax.spines['top'].set_linestyle((0, (5, 10)))
    ax.spines['bottom'].set_linestyle((0, (5, 10)))

    # Add labels and title
    ax.set_ylabel('NDCG@10')

    plt.savefig('NDCG10_negative.png')
    # Show the plot
    plt.show()
def hua_MVKG_dropout():
    # Data for AUC
    auc_data = [(0.0, 0.6621), (0.1, 0.6668), (0.2, 0.6690), (0.3, 0.6645), (0.4, 0.6612), (0.5, 0.6557), (0.6, 0.6528)]

    # Data for MRR
    mrr_data = [(0.0, 0.3046), (0.1, 0.308), (0.2, 0.3137), (0.3, 0.3087), (0.4, 0.3055), (0.5, 0.3039), (0.6, 0.3027)]

    # Data for NDCG@5
    ndcg5_data = [(0.0, 0.3418), (0.1, 0.3465), (0.2, 0.3515), (0.3, 0.3483), (0.4, 0.3432), (0.5, 0.3411),
                  (0.6, 0.3385)]

    # Data for NDCG@10
    ndcg10_data = [(0.0, 0.4018), (0.1, 0.4063), (0.2, 0.4108), (0.3, 0.4068), (0.4, 0.4031), (0.5, 0.4001),
                   (0.6, 0.3983)]

    # Extract x and y values for each metric
    auc_x, auc_y = zip(*auc_data)
    mrr_x, mrr_y = zip(*mrr_data)
    ndcg5_x, ndcg5_y = zip(*ndcg5_data)
    ndcg10_x, ndcg10_y = zip(*ndcg10_data)
    # Set global font size
    plt.rcParams.update({'font.size': 14})  # 设置全局字体大小为14
    # Plotting the line graphs with different marker styles and without connecting lines
    plt.plot(auc_x, auc_y, marker='o', linestyle='-', markeredgecolor='blue', label='AUC')
    plt.plot(mrr_x, mrr_y, marker='*', linestyle='-', markeredgecolor='orange', label='MRR')
    plt.plot(ndcg5_x, ndcg5_y, marker='^', linestyle='-', markeredgecolor='green', label='NDCG@5')
    plt.plot(ndcg10_x, ndcg10_y, marker='p', linestyle='-', markeredgecolor='red', label='NDCG@10')

    # Set the x-axis limits and ticks
    plt.xlim(-0.05, 0.65)
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # Set the y-axis limits and ticks
    plt.ylim(0.27, 0.7)
    plt.yticks([0.29, 0.32, 0.36, 0.39, 0.42, 0.64, 0.68])

    # Draw horizontal lines to divide the y-axis into 4 segments
    plt.hlines(y=[0.32, 0.36, 0.42, 0.68], xmin=-0.05, xmax=0.65, color='gray', linestyle='--')

    # Remove gridlines
    plt.grid(axis='y')  # 只保留横向网格线
    plt.gca().yaxis.grid(linestyle='--')  # 将横向网格线改为虚线

    # Add legend
    plt.legend()

    # Add labels and title
    plt.xlabel('Mask probability p')
    # Adjust layout to make room for the x-axis label
    plt.subplots_adjust(bottom=0.15)  # 调整底部边距

    # Automatically adjust subplot parameters to give specified padding
    plt.tight_layout()

    # Show the plot
    plt.show()
if __name__ == '__main__':

    # hua_MVKG_dropout()
    hua_LSMV_negative_AUC()
    hua_LSMV_negative_MRR()
    hua_LSMV_negative_NDCG5()
    hua_LSMV_negative_NDCG10()