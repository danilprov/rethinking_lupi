from matplotlib import pyplot as plt
import pickle
import numpy as np

label_map = {0.: 'priv', -1.: 'No PI', -2.: 'TRAM', -3.0: 'Generalized Distillation'}

path = 'logs/final_submission/'
with open(path + '/best_metrics.pickle', 'rb') as handle:
    best_metrics = pickle.load(handle)
labels, data = [*zip(*best_metrics.items())]  # 'transpose' items to parallel key, value lists

labels, data = best_metrics.keys(), best_metrics.values()

for label, elem in zip(labels, data):
    print(label_map[label], ': ',
          round(np.mean(elem), 4),
          round(np.std(elem), 4))

plt.boxplot(data)
plt.xticks(range(1, len(labels) + 1), labels)
plt.show()

path = 'logs/final_submission/'
with open(path + '/best_val_losses.pickle', 'rb') as handle:
    best_val_losses = pickle.load(handle)
labels, data = [*zip(*best_val_losses.items())]  # 'transpose' items to parallel key, value lists

labels, data = best_val_losses.keys(), best_val_losses.values()

for label, elem in zip(labels, data):
    print(label_map[label], ': ',
          round(np.mean(elem), 4),
          round(np.std(elem), 4))

plt.boxplot(data)
plt.xticks(range(1, len(labels) + 1), labels)
plt.show()