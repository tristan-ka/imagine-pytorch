import click
import pickle
import json
import sys

import numpy as np

sys.path.append('../../../')
from src.utils.util import set_global_seeds
import src.architecture_le2.experiment.config as config
from src.architecture_le2.interaction import RolloutWorker
from src.architecture_le2.goal_sampler import GoalSampler
from src.playground_env.reward_function import get_reward_from_state
from src.architecture_le2.goal_generator.descriptions import get_descriptions
from src.playground_env.env_params import ENV_ID

PATH = '/path_to_trial_id/trial_id/'
EPOCH = 160
POLICY_FILE = PATH + 'policy_checkpoints/policy_{}.pkl'.format(EPOCH)
PARAMS_FILE = PATH + 'params.json'

@click.command()
@click.argument('policy_file', type=str, default=POLICY_FILE)
@click.option('--seed', type=int, default=int(np.random.randint(1e6)))
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
def main(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load params
    with open(PARAMS_FILE) as json_file:
        params = json.load(json_file)

    if not render:
        env = 'PlaygroundNavigation-v1'
    else:
        env = 'PlaygroundNavigationRender-v1'
    params, rank_seed = config.configure_everything(rank=0,
                                                    seed=seed,
                                                    num_cpu=params['experiment_params']['n_cpus'],
                                                    env=env,
                                                    trial_id=0,
                                                    n_epochs=10,
                                                    reward_function=params['conditions']['reward_function'],
                                                    policy_encoding=params['conditions']['policy_encoding'],
                                                    bias_buffer=params['conditions']['bias_buffer'],
                                                    feedback_strategy=params['conditions']['feedback_strategy'],
                                                    policy_architecture=params['conditions']['policy_architecture'],
                                                    goal_invention=params['conditions']['goal_invention'],
                                                    reward_checkpoint=params['conditions']['reward_checkpoint'],
                                                    rl_positive_ratio=params['conditions']['rl_positive_ratio'],
                                                    p_partner_availability=params['conditions']['p_social_partner_availability'],
                                                    git_commit='')

    policy_language_model, reward_language_model = config.get_language_models(params)

    onehot_encoder = config.get_one_hot_encoder()
    goal_sampler = GoalSampler(policy_language_model=policy_language_model,
                               reward_language_model=reward_language_model,
                               goal_dim=policy_language_model.goal_dim,
                               one_hot_encoder=onehot_encoder,
                               **params['goal_sampler'],
                               params=params)
    reward_function = config.get_reward_function(goal_sampler, params)
    if params['conditions']['reward_function'] == 'learned_lstm':
        reward_function.restore_from_checkpoint(PATH + 'reward_checkpoints/reward_func_checkpoint_{}'.format(EPOCH))
    policy_language_model.set_reward_function(reward_function)
    if reward_language_model is not None:
        reward_language_model.set_reward_function(reward_function)
    goal_sampler.update_discovered_goals(params['all_descriptions'], episode_count=0, epoch=0)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)

    evaluation_worker = RolloutWorker(make_env=params['make_env'],
                                      policy=policy,
                                      reward_function=reward_function,
                                      params=params,
                                      render=render,
                                      **params['evaluation_rollout_params'])
    evaluation_worker.seed(seed)

    # Run evaluation.
    evaluation_worker.clear_history()

    _, test_descriptions, _ = get_descriptions(ENV_ID)

    successes_test_descr = []
    for d in test_descriptions:
        successes_test_descr.append([])
        print(d)
        for i in range(n_test_rollouts):
            goal_str = [d]
            goal_encoding = [policy_language_model.encode(goal_str[0])]
            goal_id = [0]
            ep = evaluation_worker.generate_rollouts(exploit=True,
                                                     imagined=False,
                                                     goals_str=goal_str,
                                                     goals_encodings=goal_encoding,
                                                     goals_ids=goal_id)
            out = get_reward_from_state(ep[0]['obs'][-1], goal_str[0])
            successes_test_descr[-1].append(out == 1)
        print('Success rate {}: {}'.format(d, np.mean(successes_test_descr[-1])))
    print('Global success rate: {}'.format(np.mean(successes_test_descr)))

if __name__ == '__main__':
    main()
