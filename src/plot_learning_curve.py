"""
This program plot the learning curve of all saved models
"""
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

results_dir = 'results/'
all_results = os.listdir(results_dir)
for result in all_results:
    if '.pickle' in result:
        model = result.split('.')[0]
        with open(results_dir + result, 'rb') as fp:
            curve = pickle.load(fp)
        plt.plot(np.arange(len(curve)), curve, label=model)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
# plt.savefig(results_dir + 'update_learning_curve.jpg')
plt.show()
