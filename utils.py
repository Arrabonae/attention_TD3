import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from config import *

def plot_learning_curve():
    x = [i+1 for i in range(len(SCORES_HISTORY))]        
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, SCORES_HISTORY, label = 'Episode reward', color= 'green')
    ax2.plot(x, SUCCESS_HISTORY, label = 'Success rate', color= 'blue')
    ax1.set_ylabel("Score")
    ax2.set_ylabel("Success rate")
    ax1.set_xlabel("Episodes")
    ax1.legend()
    plt.title('Performance of the agents')
    plt.savefig(FIGURE_FILE)
    plt.clf()

    x = [i+1 for i in range(len(CRITIC_LOSS1))]  
    _, ax3 = plt.subplots()
    ax3.plot(x, CRITIC_LOSS1, label='Critic loss 1', color='blue')
    ax3.plot(x, CRITIC_LOSS2, label='Critic loss 2', color='green')
    ax3.set_ylabel("Critics Loss")
    ax3.set_xlabel("Update cycles")
    plt.title('Overall Critic Loss')
    ax3.legend()
    plt.savefig(FIGURE_FILE2)
    plt.clf()

    x = [i+1 for i in range(len(ACTOR_LOSS))]  
    _, ax4 = plt.subplots()
    ax4.plot(x, ACTOR_LOSS, label='Actor loss 1', color='red')
    ax4.set_ylabel("Actor Loss")
    ax4.set_xlabel("Update cycles")
    plt.title('Overall Actor Loss')
    ax4.legend()
    plt.savefig(FIGURE_FILE3)


class OrnsteinUhlenbeckActionNoise():
    """
    OpenAI baselines implementation of Ornstein-Uhlenbeck process
    """
    def __init__(self, mu):
        self.theta = THETA
        self.mu = mu
        self.sigma = SIGMA
        self.dt = DT
        self.x0 = X0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)