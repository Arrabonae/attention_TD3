#General
EPOCHS = 10
CYCLES = 50
EPISODES_PER_CYCLE = 16
OPTIMIZER_STEPS = 40
N_EVAL = 10
FIGURE_FILE = 'plots/learning_curve.png'
FIGURE_FILE2 = 'plots/critic_loss.png'
FIGURE_FILE3 = 'plots/actor_loss.png'
CHECKPOINT_DIR = 'models/'
SCORES_HISTORY = []
SUCCESS_HISTORY = []    
CRITIC_LOSS1 = []
CRITIC_LOSS2 = []
ACTOR_LOSS = []

#Memory
BATCH_SIZE = 256
BUFFER_SIZE = 10**6
T = 50 #size of one episode
REPLAY_K = 4

#Training
ALPHA = 0.001
BETA = 0.0005
TAU = 0.001
GAMMA = 0.99

DELAY_STEPS = 3
GRACE_PERIOD = 1000


#Network architecture
WEIGHT_INIT = 'he_normal'
BIAS_INIT = 'he_normal'
CRITIC_DENSE1 = 256
CRITIC_DENSE2 = 256
CRITIC_DENSE3 = 256
ACTORS_DENSE1 = 256
ACTORS_DENSE2 = 256
ACTORS_DENSE3 = 256

CRITIC_ACTIVATION_HIDDEN = 'leaky_relu'
CRITIC_ACTIVATION_OUTPUT = None
ACTORS_ACTIVATION_HIDDEN = 'leaky_relu'
ACTORS_ACTIVATION_OUTPUT = 'tanh' #Action low and high are -1 and 1

#Ornstein-Uhlenbeck process
THETA = 0.15
SIGMA = 0.2
DT = 1e-2
X0 = None