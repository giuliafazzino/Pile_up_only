"""
Plot true energy vs reconstructed energy before and after the cut
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Load data 
    dir_path = 'out'
    plot_path = dir_path + '/plots'

    data = pd.read_csv('data/df_test.csv')
    #probs = data['pred'].to_numpy()

    threshold = .5
    no_pu = data['pred'] < threshold

    # Plot true response and pred response
    plt.figure(figsize = (10,5))
    bins = np.linspace(0, 10, 100)
    plt.hist(data['r_e_calculated'], bins, histtype = 'step', color = 'red', label = 'Full dataset')
    plt.hist(data[no_pu]['r_e_calculated'], bins, histtype = 'step', color = 'blue', label = 'Without PU-only')
    #plt.text(3, 1500, f'Median true response = {np.median(resp_test):.2f}')
    #plt.text(3, 700, f'Mean true response = {np.mean(resp_test):.2f}')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Trueesponse')
    plt.ylabel('Events')
    plt.savefig(plot_path + '/response_test.png')


if __name__ == "__main__":
    main()