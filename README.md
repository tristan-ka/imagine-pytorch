# Welcome to the IMAGINE repository

## Requirements

The dependencies are listed in the requirements.txt file. Our conda environment can be cloned with:
```
conda env create -f environment.yml
```

## Supervised learning of the reward function

In order to perform the experiment of the paper follow the following instructions:

1. Generate trajectories by interacting with the environment

Go to playground_env directory and run:
```
python dataset_generator.py
```
2. Split between train and test set

Go to the supervised_learning directory and run:
```
python create_datasets.py
```

3. Training the reward function

In the same directory run:
```
python train.py --architecture=modular_attention --n_epochs=10 --positive_ratio=0.1 --dataset=processed --trial_id=1
```


## RL training

1. Running the algorithm
The main running script is /src/architecture_le2/experiments/train.py. It can be used as such:

```
python train.py --num_cpu=6 --architecture=modular_attention --reward_function=learned_lstm  --goal_invention=from_epoch_80 --n_epochs=167
```

Note that the number of cpu is an important parameter. Changing it is **not** equivalent to reducing/increasing training time. One epoch is 600 episodes. Other parameters can be
 found in train.py. The config.py file contains all parameters and is overriden by parameters defined in train.py.
 
 Logs and results are saved in /src/data/expe/PlaygroundNavigation-v1/trial_id/. It contains policy and reward function checkpoints, raw logs (log.txt), a csv containing main metrics (progress.csv) and a json file with the parameters (params.json).
 
 2. Plotting results
 Results for one run can be plotted using the script /src/analyses/new_plot.py