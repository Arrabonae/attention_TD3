import gymnasium as gym
import panda_gym

#we use this environment as this has a sparse reward function which helps us demonstrate HER. 
env = gym.make("PandaPush-v3")
observation, info = env.reset()

print(env.observation_space)
#Dict('achieved_goal': Box(-10.0, 10.0, (3,), float32), 'desired_goal': Box(-10.0, 10.0, (3,), float32), 'observation': Box(-10.0, 10.0, (18,), float32))
print(env.action_space)
#Box(-1.0, 1.0, (3,), float32)


for _ in range(10):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print(action)
    #contninous action we need to give value for each joint between -2 <-> 2
    #Action: [ 0.72488785 -0.7759044  -0.5041433 ]
    
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward, terminated, truncated, info)
    #{'observation': array([ 5.47938906e-02,  6.45167893e-03,  1.77058756e-01, -1.48582861e-01,
    #    6.28416166e-02, -1.37010586e+00, -2.38760710e-02,  2.47766189e-02,
    #    1.99895296e-02,  4.48686751e-06, -3.08392118e-05, -3.87872387e-06,
    #   -5.73867237e-06,  1.09538505e-05, -5.08076300e-06,  5.10965457e-08,
    #   -2.53998151e-04, -9.67000524e-05], dtype=float32), 'achieved_goal': array([-0.02387607,  0.02477662,  0.01998953], dtype=float32), 'desired_goal': array([-0.05123958,  0.13161023,  0.02      ], dtype=float32)}
    # reward: -1.0 
    # terminated: False 
    # truncated: False 
    # info {'is_success': False}

    print(env.compute_reward(observation['achieved_goal'], observation['desired_goal'], dict()))
    #-1.0
    if terminated or truncated:
        observation, info = env.reset()

env.close()