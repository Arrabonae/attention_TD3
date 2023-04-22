import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
#locals
from networks import ActorNetwork, CriticNetwork
from config import *
from utils import OrnsteinUhlenbeckActionNoise


class TD3:

    def __init__(self, n_actions, actions_low, actions_high, input_shape):

        self.n_actions = n_actions
        self.gamma = GAMMA
        self.input_shape = input_shape
        #for clipping the target (per HER paper)
        self.limit = -1 / (1 - self.gamma)
        self.learn_step_cntr = 0

        self.actor = ActorAgent(n_actions, actions_low, actions_high, alpha=ALPHA)
        self.critic = CriticAgent(beta=BETA)

    def save_checkpoint(self):
        print('... saving checkpoint ...')

        self.actor.save_models()
        self.critic.save_models()

    def load_checkpoint(self):
        """
        Mo need to load critic agent, as it is only used for learning
        """
        print('... loading checkpoint ...')
        self.actor.load_models(self.input_shape)

    def update_network_parameters(self, tau):
        self.actor.update_network_parameters(tau)
        self.critic.update_network_parameters(tau)

    def choose_action(self, obs_goal, evaluate):
        """
        Environment takes parallel actions, so we need to return a dictionary of agents in the environment
        return such that {"agent_0": [action_0, action_1 ... action_n], "agent_1": [action_0, action_1 ... action_n], ...}
        """
        return self.actor.choose_action(obs_goal, evaluate)

    def learn(self, memory):

        if memory.mem_cntr < BATCH_SIZE:
            return

        states, actions, rewards, states_, dones, desired_goals = memory.sample_memory()

        states = np.concatenate([states, desired_goals], axis=1)
        states_ = np.concatenate([states_, desired_goals], axis=1)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:

            new_pi = self.actor.target_actor(states_)
            new_pi += self.actor.noise()
            new_pi = tf.clip_by_value(new_pi, self.actor.actions_low, self.actor.actions_high)

            critic_value1_ = tf.squeeze(self.critic.target_critic_1((states_, new_pi)),1)
            critic_value2_ = tf.squeeze(self.critic.target_critic_2((states_, new_pi)),1)

            #need to set it here as this can impact the minimum calculation later on (if done, then value is 0)
            critic_value1_ = critic_value1_ * (1 - dones)
            critic_value2_ = critic_value2_ * (1 - dones)
            critic_value_ = tf.reduce_min((critic_value1_, critic_value2_), axis=0)

            critic_value1 = tf.squeeze(self.critic.critic_1((states, actions)), 1)
            critic_value2 = tf.squeeze(self.critic.critic_2((states, actions)), 1)
            
            target = rewards + self.gamma * critic_value_ #* (1 - dones)
            #target = tf.reduce_mean(target, axis=0)
            target = tf.clip_by_value(target, self.limit, 0)

            #OpenAI implementation uses huber loss, but after testing, MSE works better
            critic_loss1 = keras.losses.MSE(target, critic_value1)
            critic_loss2 = keras.losses.MSE(target, critic_value2)

            critic_loss = critic_loss1 + critic_loss2


        critic_network_gradient1 = tape.gradient(critic_loss, self.critic.critic_1.trainable_variables)
        critic_network_gradient2 = tape.gradient(critic_loss, self.critic.critic_2.trainable_variables)

        self.critic.critic_1.optimizer.apply_gradients(zip(critic_network_gradient1, self.critic.critic_1.trainable_variables))
        self.critic.critic_2.optimizer.apply_gradients(zip(critic_network_gradient2, self.critic.critic_2.trainable_variables))
        
        del tape
        self.learn_step_cntr += 1
        CRITIC_LOSS1.append(critic_loss1.numpy())
        CRITIC_LOSS2.append(critic_loss2.numpy())

        if self.learn_step_cntr % DELAY_STEPS != 0:
            return

        with tf.GradientTape() as tape:
            pi = self.actor.actor(states)
            actor_loss = -tf.squeeze(self.critic.critic_1((states, pi)),1)
            actor_loss = tf.reduce_mean(actor_loss, axis=0)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.actor.trainable_variables)
        self.actor.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.actor.trainable_variables))
        
        ACTOR_LOSS.append(actor_loss.numpy())
        self.update_network_parameters(TAU) 

        return

class ActorAgent:
    """
    Class of the actor agent
    """
    def __init__(self, n_actions, actions_low, actions_high, alpha):
        
        self.n_actions = n_actions
        #for action selection: clipping / scaling the action to be between low and high
        self.actions_low = actions_low
        self.actions_high = actions_high
        self.time_step = 0

        #per OpenAI paper, Ornstein-Uhlenbeck process for action noise is the best to introduce exploration
        self.noise = OrnsteinUhlenbeckActionNoise(mu= np.zeros(self.n_actions))

        self.actor = ActorNetwork(n_actions=n_actions, name= 'Actor')
        self.target_actor = ActorNetwork(n_actions=n_actions, name= 'Target_actor')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        if targets == []:
            return
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        if weights == []:
            return 
        self.target_actor.set_weights(weights)

    def choose_action(self, obs, evaluate):

        if self.time_step < GRACE_PERIOD:
            self.time_step += 1
            return np.random.uniform(self.actions_low, self.actions_high, self.n_actions)

        state = tf.convert_to_tensor([obs], dtype=tf.float32)
        actions = self.actor(state)
        #add noise for exploration
        if not evaluate:
            actions += tf.convert_to_tensor(self.noise(), dtype=tf.float32)
        #clip the action to be between low and high otherwise environment will do it for you but 
        #it will affect performance and gives warning message
        actions = tf.clip_by_value(actions, self.actions_low, self.actions_high)
        self.time_step += 1
        return actions[0].numpy()

    def save_models(self):
        #print('... saving {} model ...' .format(self.actor.model_name))
        self.actor.save_weights(self.actor.checkpoint_file, save_format='h5')
        #print('... saving {} model ...' .format(self.target_actor.model_name))
        self.target_actor.save_weights(self.target_actor.checkpoint_file, save_format='h5')


    def load_models(self, actor_shape):
        print('... loading {} model ...'.format(self.actor.model_name))
        self.actor.build((BATCH_SIZE, actor_shape))
        self.actor.load_weights(self.actor.checkpoint_file)
        print('... loading {} model ...' .format(self.target_actor.model_name))
        self.target_actor.build((BATCH_SIZE, actor_shape))
        self.target_actor.load_weights(self.target_actor.checkpoint_file)


class CriticAgent():
    """
    Class of the critic agent
    """
    def __init__(self, beta):

        self.critic_1 = CriticNetwork(name='Critic_1')
        self.critic_2 = CriticNetwork(name='Critic_2')
        self.target_critic_1 = CriticNetwork(name='Target_critic_1')
        self.target_critic_2 = CriticNetwork(name='Target_critic_2')

        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        #update first pair of target critic networks
        weights = []
        targets = self.target_critic_1.weights
        if targets == []:
            return
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic_1.set_weights(weights)

        #update second pair of target critic networks
        weights = []
        targets = self.target_critic_2.weights
        if targets == []:
            return
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic_2.set_weights(weights)

    def save_models(self):
        self.critic_1.save_weights(self.critic_1.checkpoint_file, save_format='h5')
        self.critic_2.save_weights(self.critic_2.checkpoint_file, save_format='h5')

        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file, save_format='h5')
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file, save_format='h5')

    def load_models(self):
        print('... loading {} model ...'.format(self.critic_1.model_name))
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        print('... loading {} model ...'.format(self.critic_2.model_name))
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        print('... loading {} and {} models ...'.format(self.target_critic_1.model_name, self.target_critic_2.model_name))
        self.update_network_parameters(TAU)