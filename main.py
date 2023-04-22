import gymnasium as gym
import panda_gym
import numpy as np
from agent import TD3
from buffer import HER
from config import *
from utils import plot_learning_curve

def train(memory, agent, env):
    best_success, best_score = -np.inf, -np.inf
    for i in range(EPOCHS):
        for _ in range(CYCLES):
            for _ in range(EPISODES_PER_CYCLE):
                _, _ = play_episode(memory, agent, env)
            for _ in range(OPTIMIZER_STEPS):
                agent.learn(memory)
            agent.update_network_parameters(TAU)
        test_score, test_success = [], []
        for _ in range(N_EVAL):
            score, success = play_episode(memory, agent, env, evaluate=True)
            test_success.append(success)
            test_score.append(score)

        if np.mean(test_score) > best_score:
            best_success = np.mean(test_success)
            best_score = np.mean(test_score)
            agent.save_checkpoint()
            print('Best score so far: {:.1f}; best success so far: {:.2%}' .format(best_score, best_success))

        SCORES_HISTORY.append(np.mean(test_score))
        SUCCESS_HISTORY.append(np.mean(test_success))
        print('Epoch: {} Score: {:.1f}; Success: {:.2%}' .format(i, np.mean(test_score), np.mean(test_success)))

def play_episode(memory, agent, env, evaluate=False):
    obs, info = env.reset()
    observation = obs['observation']
    achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    done = False
    score = 0
    states, actions, rewards, states_, dones, d_goal, a_goal, a_goal_, infos = [], [], [], [], [], [], [], [], []
    
    while not done:

        action = agent.choose_action(np.concatenate([observation, desired_goal]), evaluate)

        observation_, reward, done, truncated, info = env.step(action)

        states.append(observation)
        states_.append(observation_['observation'])
        rewards.append(reward)
        actions.append(action)
        dones.append(done)
        d_goal.append(desired_goal)
        a_goal.append(achieved_goal)
        a_goal_.append(observation_['achieved_goal'])
        infos.append(info['is_success'])
        
        score += reward
        achieved_goal = observation_['achieved_goal']
        observation = observation_['observation']
    
        if truncated:
            done = True

    if not evaluate:
        #Openai HER
        memory.store_transition(states, actions, rewards, states_, dones, d_goal, a_goal, a_goal_, infos)
    
    success = info['is_success']

    return score, success


if __name__ == '__main__':
    env = gym.make("PandaReach-v3")
    obs_shape = env.observation_space['observation'].shape[0]
    goal_shape = env.observation_space['achieved_goal'].shape[0]
    n_actions=env.action_space.shape[0]
    
    memory = HER(obs_shape, n_actions, goal_shape, env.compute_reward)
    agent = TD3(n_actions, env.action_space.low, env.action_space.high, [obs_shape+goal_shape])

    train(memory, agent, env)
    plot_learning_curve()


