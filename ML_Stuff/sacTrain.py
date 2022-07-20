"""
Main function for training Grid Game network.
"""
# Imports.
import json
import numpy as np
import os
import yaml

# Custom imports.
from RL.SAC_V2.her_agent import Agent
from RL.SAC_V2.grid_image_env import GridImageEnv
from RL.SAC_V2.plots import plot_history


def view_model_filters(model_path, image, output_dir=''):
    sv_elems = 2
    render_elems = 3
    input_shape = np.squeeze(image[0]).shape
    n_actions = sv_elems
    input_type = 'image_diff'
    asymmetric = True

    agent = Agent(
        input_shape=input_shape,
        temperature=temperature,
        n_actions=sv_elems,
        asymmetric=asymmetric,
    )

    if model_path is not None:
        agent.load_actor(model_path)

    agent.actor.view_filters(image, output_dir=output_dir)

if __name__ == '__main__':

    cwd = os.getcwd()

    with open(f'{cwd}/sac_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # CONFIG.
    learning = True
    load_checkpoint = False
    step_size_schedule = False
    mae_step_threshold = 0.01
    input_type = 'image_diff'
    start_pos = 'random'
    input_shape = (64, 64, 3)
    sv_elems = 2
    render_elems = 3
    n_actions = sv_elems
    batch_size = 128
    num_moves = 50
    max_step = 0.1
    mae_threshold = 0.02
    actor_lr = 3e-4
    critic_lr = 3e-4
    gamma = 0.99
    temperature = 1.0
    max_size = 50000
    tau = 0.005
    reward_scale = False
    n_games = 5000
    strategy = 'future'
    her_ratio = 0.8
    asymmetric = True
    hidden_kernel = False
    hidden_bias = False
    output_kernel = True
    output_bias = False
    skip = True
    activation = 'tanh'
    
    # Feedback.
    print_env_info = False
    print_agent_info = False
    render_bool = True

    # Pathing
    home_directory = os.path.expanduser('~')
    dataset_dir = f'{home_directory}/Datasets/RL'
    output_path = f'{dataset_dir}/Output'
    weights_path = f'{output_path}'
    blend_sources = [f'{home_directory}/Datasets/blend_files/head_reduced.blend']

    env = GridImageEnv(
        sv_elems=sv_elems,
        img_shape=input_shape,
        render_elems=render_elems,
        num_moves=num_moves,
        max_step=max_step,
        start_pos=start_pos,
        mae_threshold=mae_threshold, 
        mae_step_threshold=mae_step_threshold,
        blend_sources=blend_sources,
        dataset_dir=dataset_dir, 
        step_size_schedule=step_size_schedule,
        render_bool=render_bool,
        reward_scale=reward_scale,
    )

    agent = Agent(
        input_shape=input_shape,
        alpha=actor_lr,
        beta=critic_lr,
        max_step=max_step,
        gamma=gamma,
        temperature=temperature,
        n_actions=sv_elems,
        max_size=max_size,
        tau=tau,
        batch_size=batch_size,
        input_type=input_type,
        strategy=strategy,
        her_ratio=her_ratio,
        hidden_kernel=hidden_kernel,
        hidden_bias=hidden_bias,
        output_kernel=output_kernel,
        output_bias=output_bias,
        asymmetric=asymmetric,
        skip=skip,
        activation=activation
    )

    best_score = -1000

    history = {
        'score': [],
        'avg_score': [],
        'num_wins': 0
    }

    counter = 0

    # Save initial feature maps.
    state, _ = env.reset()
    print('Visualizing initial feature maps...')
    view_model_filters(model_path=None, image=state,
        output_dir=f'{output_path}/initial_feature_maps')

    print('Begin training...')
    for step in range(n_games):
        state, svs = env.reset()
        ep_reward_sum = 0
        episode = 1
        done = False
        episode_counter = 0

        while not done:
            counter += 1
            episode_counter += 1
            
            action, mus, vars = agent.choose_action(state)
            new_state, new_svs, reward, done, info = env.step(action)

            if episode_counter > 1:
                agent.remember(state, action, reward, new_state, done, svs, new_svs)
            elif done:
                counter -= 1

            ep_reward_sum += reward
            state = new_state

            if learning:
                agent.learn()

            if render_bool:
                env.render(mus, vars)

        
        print(f'End Game {step + 1}! Final Score: {ep_reward_sum}')
    
        history['score'].append(ep_reward_sum)
        history['avg_score'].append(np.mean(history['score'][-100:]))
        if info['win']:
            history['num_wins'] += 1

        with open(f'{output_path}/history.json', 'w') as file:
            json.dump(history, file)
        
        if history['avg_score'][-1] > best_score and episode > 10:
            agent.save_models()
            best_score = history['avg_score'][-1]
        
        ep_reward_sum = 0
        episode += 1

    image, _ = env.reset()
    view_model_filters(
        model_path=f'{output_path}/actor.h5',
        image=image,
        output_dir=f'{output_path}/feature_maps'
    )
    
    plot_history(f'{home_directory}/Datasets/RL/Output/history.json',
        out_filepath=output_path)