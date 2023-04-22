import numpy as np
from config import *


class HER:
    def __init__(self, input_shape, n_actions, goal_shape, compute_reward):
        self.memory_size = int(BUFFER_SIZE / T)
        self.batch_size = BATCH_SIZE
        self.input_shape = input_shape
        self.compute_reward = compute_reward
        self.n_actions = n_actions
        self.goal_shape = goal_shape
        self.mem_cntr = 0

        self.states = np.zeros((self.memory_size, T, self.input_shape),dtype=np.float32)
        self.states_ = np.zeros((self.memory_size, T, self.input_shape), dtype=np.float64)
        self.actions = np.zeros((self.memory_size, T, self.n_actions), dtype=np.float32)
        self.rewards = np.zeros([self.memory_size, T], dtype=np.float32)
        self.dones = np.zeros([self.memory_size, T], dtype=np.bool_)
        
        self.desired_goals = np.zeros((self.memory_size, T, self.goal_shape), dtype=np.float32)
        self.achieved_goals = np.zeros((self.memory_size, T,  self.goal_shape), dtype=np.float32)
        self.achieved_goals_ = np.zeros((self.memory_size, T, self.goal_shape), dtype=np.float64)
        self.infos = np.zeros((self.memory_size, T), dtype=np.bool_)
        print("Running with HER")

    def store_transition(self, state, action, reward, state_, done, d_goal, a_goal, a_goal_, infos):

        i = self.mem_cntr % self.memory_size

        #if episode was shorted than T, pad with last state
        for _ in range(T-len(state)):
            state.append(state[-1])
            state_.append(state_[-1])
            action.append(action[-1])
            reward.append(reward[-1])
            done.append(done[-1])
            d_goal.append(d_goal[-1])
            a_goal.append(a_goal[-1])
            a_goal_.append(a_goal_[-1])
            infos.append(infos[-1])
        
        assert(len(state) == T)
        assert(len(state_) == T)
        assert(len(action) == T)
        assert(len(reward) == T)
        assert(len(done) == T)
        assert(len(d_goal) == T)
        assert(len(a_goal) == T)
        assert(len(a_goal_) == T)
        assert(len(infos) == T)

        self.states[i] = state
        self.states_[i] = state_
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.desired_goals[i] = d_goal
        self.achieved_goals[i] = a_goal
        self.achieved_goals_[i] = a_goal_
        self.infos[i] = infos
        
        self.mem_cntr += 1

    def sample_memory(self):
        """
        Per OpenAI baselines
        I've adopted future strategy with k=4
        """
        future_p = 1 - (1. / (1 + REPLAY_K))

        # Select which episodes and time steps to use.  
        max = min(self.mem_cntr, self.memory_size)
        episode_samples = np.random.randint(0, max, self.batch_size)
        t_samples = np.random.randint(T, size=self.batch_size)

        sample_states = self.states[episode_samples, t_samples]
        sample_actions = self.actions[episode_samples, t_samples]
        sample_rewards = self.rewards[episode_samples, t_samples]
        sample_states_ = self.states_[episode_samples, t_samples]
        sample_dones = self.dones[episode_samples, t_samples]
        sample_desired_goals = self.desired_goals[episode_samples, t_samples]
        sample_achieved_goals = self.achieved_goals[episode_samples, t_samples]
        sample_infos = self.infos[episode_samples, t_samples]


        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=self.batch_size) < future_p)
        future_offset = np.random.uniform(size=self.batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.

        future_achieved_goal =  self.achieved_goals[episode_samples[her_indexes], future_t]
        sample_desired_goals[her_indexes] = future_achieved_goal
        future_infos = self.infos[episode_samples[her_indexes], future_t]
        sample_infos[her_indexes] = future_infos

        #  Re-compute reward since we may have substituted the goal.
        for idx, value in enumerate(sample_infos):
            sample_rewards[idx] = self.compute_reward(sample_achieved_goals[idx], sample_desired_goals[idx], {'is_success': value}) #'is_success': value

        assert(sample_states.shape == (self.batch_size, self.input_shape))
        assert(sample_actions.shape == (self.batch_size, self.n_actions))
        assert(sample_rewards.shape == (self.batch_size, ))
        assert(sample_states_.shape == (self.batch_size, self.input_shape))
        assert(sample_dones.shape == (self.batch_size, ))
        assert(sample_desired_goals.shape == (self.batch_size, self.goal_shape))

        return sample_states, sample_actions, sample_rewards, sample_states_, sample_dones, sample_desired_goals