import json
import matplotlib.pyplot as plt
import numpy as np

def plot_history(json_filepath, out_filepath='.', exp_name=None):

    f = open(json_filepath)

    data = json.load(f)
    
    num_plots = len(data.keys()) - 1
    
    plt.figure(figsize=(40,10))

    if exp_name:
        plt.suptitle(exp_name)
    for i, key in enumerate(data.keys()):
        if key != 'win_results':
            plt.subplot(1, num_plots, 1 + i)
            plt.plot(data[key], color='blue', alpha=1.0)
            plt.title(key)
            plt.legend()
    plt.savefig(f'{out_filepath}/history.png')
    plt.close()

if __name__ == '__main__':

    json_filepath = '/home/askoch/Datasets/RL/Output/history.json'
    out_filepath = '/home/askoch/Datasets/RL/Output'

    plot_history(
        json_filepath=json_filepath,
        out_filepath=out_filepath,
        exp_name='Head Game (2D)')