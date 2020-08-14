import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from src.playground_env.env_params import thing_colors, plants

font = {'size': 25}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.494, 0.1844, 0.556], [0, 0.447, 0.7410], [0.3010, 0.745, 0.933], [0.85, 0.325, 0.098],
          [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184]]

plants.remove('flower')
types_words = [['Grasp red tree', 'Grasp blue door', 'Grasp green dog', 'Grow green dog'],
               ['Grasp any flower', 'Grasp red flower', 'Grasp blue flower', 'Grasp green flower', 'Grow any flower',
                'Grow red flower', 'Grow blue flower', 'Grow green flower'],
               ['Grasp {} animal'.format(d) for d in thing_colors + ['any']],
               ['Grasp {} fly'.format(d) for d in thing_colors + ['any']],
               ['Grow {} {}'.format(c, p) for c in thing_colors + ['any'] for p in plants + ['plant', 'living_thing']]
               ]
types_words[4].remove('Grow red tree')
type_legends = ['Type {}'.format(i) for i in range(1, len(types_words) + 1)]
n_types = len(type_legends)

output_dir = '../../data/output/'
pytorch_exp = '100'
tf_exp = '10000'

with open(output_dir + '{}/id2description.pk'.format(pytorch_exp), 'rb') as fp:
    id2description = pickle.load(fp)
with open(output_dir + '{}/loss_log.pk'.format(pytorch_exp), 'rb') as fp:
    pytorch_loss = pickle.load(fp)
with open(output_dir + '{}/loss_log.pk'.format(tf_exp), 'rb') as fp:
    tf_loss = pickle.load(fp)
with open(output_dir + '{}/f1_states.pk'.format(pytorch_exp), 'rb') as fp:
    f1_states_pytorch = pickle.load(fp)
with open(output_dir + '{}/f1_states.pk'.format(tf_exp), 'rb') as fp:
    f1_states_tf = pickle.load(fp)
with open(output_dir + '{}/f1_language.pk'.format(pytorch_exp), 'rb') as fp:
    f1_language_pytorch = pickle.load(fp)
with open(output_dir + '{}/f1_language.pk'.format(tf_exp), 'rb') as fp:
    f1_language_tf = pickle.load(fp)

types_ids = []
for types in types_words:
    ids = []
    for id, descr in id2description.items():
        if descr in types:
            ids.append(id)
    types_ids.append(ids)

fig = plt.figure(figsize=(16, 9))
plt.plot(pytorch_loss, linewidth=5, label='Pytorch')
plt.plot(tf_loss, linewidth=5, label='Tensorflow')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Binary Cross Entropy Loss')
plt.legend()


def compute_f1_per_type(f1_dict, types_ids, N):
    mean_f1_types = []
    std_f1_types = []
    for type_ids in types_ids:
        mean_f1s = []
        std_f1s = []
        for n_iter in range(N):
            mean_f1s.append(np.nanmean([f1_dict[i][n_iter] for i in type_ids]))
            std_f1s.append(np.nanstd([f1_dict[i][n_iter] for i in type_ids]))
        mean_f1_types.append(mean_f1s)
        std_f1_types.append(std_f1s)

    return mean_f1_types, std_f1_types


a, b = compute_f1_per_type(f1_language_pytorch, types_ids, 5)
c, d = compute_f1_per_type(f1_language_tf, types_ids, 5)

for type, mean_type_pytorch, std_type_pytroch, mean_type_tf, std_type_tf in zip(type_legends, a, b, c, d):
    mean_pyt = np.nan_to_num(mean_type_pytorch)
    mean_tf = np.nan_to_num(mean_type_tf)
    std_pyt = np.nan_to_num(std_type_pytroch)
    std_tf = np.nan_to_num(std_type_tf)
    plt.figure(figsize=(16, 9))
    plt.title('Language Generalization: ' + type)
    plt.plot([50 * i for i in range(5)], mean_pyt, linewidth=5, label='Pytorch', color=colors[0])
    plt.plot([50 * i for i in range(5)], mean_tf, linewidth=5, label='Tensorflow', color=colors[1])
    plt.fill_between([50 * i for i in range(5)], mean_pyt - std_pyt, mean_pyt + std_pyt, color=colors[0], alpha=0.2)
    plt.fill_between([50 * i for i in range(5)], mean_tf - std_tf, mean_tf + std_tf, color=colors[1], alpha=0.2)
    plt.grid()
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('$F_1$-score')


def compute_f1_mean_std(f1_dict, N):
    mean_f1 = []
    std_f1 = []
    for n_iter in range(N):
        mean_f1.append(np.nanmean([f1_dict[i][n_iter] for i in f1_dict.keys()]))
        std_f1.append(np.nanstd([f1_dict[i][n_iter] for i in f1_dict.keys()]))

    return np.array(mean_f1), np.array(std_f1)


a, b = compute_f1_mean_std(f1_states_pytorch, 5)
c, d = compute_f1_mean_std(f1_states_tf, 5)

fig = plt.figure(figsize=(16, 9))
plt.plot([50 * i for i in range(5)], a, linewidth=5, label='Pytorch', color=colors[0])
plt.plot([50 * i for i in range(5)], c, linewidth=5, label='Tensorflow', color=colors[1])
plt.fill_between([50 * i for i in range(5)], a - b, a + b, color=colors[0], alpha=0.2)
plt.fill_between([50 * i for i in range(5)], c - d, c + d, color=colors[1], alpha=0.2)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('$F_1$-score')
plt.title('State Generalization')
plt.show()
stop = 1
