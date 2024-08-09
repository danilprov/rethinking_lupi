import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.ticker import FormatStrFormatter


class_task = 'classification_data/logs/final_submission'
bandit_task = 'bandit_data/logs/final_submission'
list_of_folders = [bandit_task + '/ijcai', class_task + '/health_data', class_task + '/nasa',
                   class_task + '/drink']
fig, axs = plt.subplots(2, len(list_of_folders), sharex='col', figsize=(15, 7))
key_map = {-3.: 'Gen. dist.', -2.: 'TRAM', -1.: 'No PI', 0.: 'Teacher'}

i = 0
for folder in list_of_folders:
    print(i, folder)
    if 'drink' in folder:
        title = 'Smoker or Drinker'
        y_min_metric, y_max_metric = 0.6, None
        y_min_loss, y_max_loss = None, 0.7
    elif 'health_data' in folder:
        title = 'Heart Disease'
        y_min_metric, y_max_metric = 0.56, 0.69
        y_min_loss, y_max_loss = 0.235, 0.28
    elif 'nasa' in folder:
        title = 'NASA-NEO'
        y_min_metric, y_max_metric = 0.88, 0.92
        y_min_loss, y_max_loss = 0.175, 0.23
    elif 'ijcai' in folder:
        title = 'Repeated Buyers'
        y_min_metric, y_max_metric = 0.56, 0.75
        y_min_loss, y_max_loss = 0.185, 0.245


    with open(folder + '/best_metrics.pickle', 'rb') as f:
        metrics = pickle.load(f)
    with open(folder + '/best_val_losses.pickle', 'rb') as f:
        val_losses = pickle.load(f)

    print(metrics.keys())

    for key in [-1., -2., -3., 0]:
        results = metrics[key]
        mean_performance = np.mean(results, axis=1)
        max_performance = np.max(results, axis=1) - mean_performance
        min_performance = mean_performance - np.min(results, axis=1)
        epochs = np.arange(results.shape[0])
        print('accuracy:', key, round(mean_performance[-1] * 100, 2), round(np.std(results, axis=1)[-1] * 100, 2))

        axs[0, i].plot(epochs, mean_performance, label=key_map[key])
        axs[0, i].fill_between(epochs, mean_performance - min_performance, mean_performance + max_performance, alpha=0.2)
        axs[0, i].grid(visible=True, which='major', linestyle='--', alpha=0.5)
        axs[0, i].minorticks_on()
        axs[0, i].grid(visible=True, which='minor', linestyle=':', alpha=0.2)
        axs[0, i].set_title(title)
        axs[0, i].set_ylim(y_min_metric, y_max_metric, auto=True)
        axs[0, i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    for key in [-1., -2., -3., 0]:
        results = val_losses[key]
        mean_performance = np.mean(results, axis=1)
        epochs = np.arange(results.shape[0])
        max_performance = np.max(results, axis=1) - mean_performance
        min_performance = mean_performance - np.min(results, axis=1)
        print('loss:', key, round(mean_performance[-1], 4), round(np.std(results, axis=1)[-1], 4))

        axs[1, i].plot(epochs, mean_performance)
        axs[1, i].fill_between(epochs, mean_performance - min_performance, mean_performance + max_performance, alpha=0.2)
        axs[1, i].set_xlabel('Training steps')
        axs[1, i].grid(visible=True, which='major', linestyle='--', alpha=0.5)
        axs[1, i].minorticks_on()
        axs[1, i].grid(visible=True, which='minor', linestyle=':', alpha=0.2)
        axs[1, i].set_ylim(y_min_loss, y_max_loss, auto=True)
        axs[1, i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    i += 1
axs[0, 0].set_ylabel('Performance metric')
axs[1, 0].set_ylabel('Cross entropy loss')

lines_labels = [ax.get_legend_handles_labels() for ax in axs[:,0]]
lines, labels = [sum(_, []) for _ in zip(*lines_labels)]
fig.legend(lines, labels,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.03),
           fancybox=True, shadow=True, ncol=5, prop={'size': 15})
plt.savefig('real_world_data_result.pdf', bbox_inches='tight')
plt.show()

