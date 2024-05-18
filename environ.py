# Important Libraries
from pytao import Tao
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import gymnasium as gym
from gymnasium import spaces
from time import time


# Extracts data
def get_data_values(tao : Tao, data_dict : dict):
  d1_datas = data_dict.keys()
  extracted_data = []
  for d1 in d1_datas:
    data = tao.cmd('show data ' + d1)
    for row in data:
      if d1 in row and data_dict[d1] in row:
        extracted_data.append(row.split()[5])
  return np.array(extracted_data, dtype = np.float64)

# Extracts variable values
def get_variable_values(tao : Tao, var_list : list):
  extracted_var_values = []
  for v1 in var_list:
    var = tao.cmd('show var ' + v1)
    for row in var:
      if v1 in row:
        extracted_var_values.append(row.split()[3])
  return np.array(extracted_var_values, dtype = np.float64)

# Sets new variable values
def set_new_variable_values(tao : Tao, var_list : list, new_values : np.ndarray):
  for v1 in var_list:
    var = tao.cmd('show var ' + v1)
    var_n = 0
    for row in var:
      i = 0
      if v1 in row:
        tao.cmd('set var '+row.split()[0]+'|model = '+str(new_values[var_n + i]))
        i += 1
      var_n += i
  return




# Environment Class
class LTBEnv(gym.Env):

  ##############################################################################
  # Initialize the Environment
  def __init__(self, init_file : str, 
               data_dict : dict, 
               var_list : list, 
               action_low : float, 
               action_high : float,
               obs_low : float = -1.0,
               obs_high : float = 1.0,
               max_steps : int = 200,
               convergence_val : float = 1e-4,
               bad_convergence_val : float = 1e-3,
               reward_alpha : float = 1e-1,
               reward_beta : float = 1e-2,
               reward_fun : int = 2):

    # Initialize a tao object
    self.init_file = init_file
    self.tao = Tao('-init ' + self.init_file)

    # Data + Var Info
    self.data_dict = data_dict # data measured that makes up your state space
    self.var_list = var_list # list of variables that you will change

    # Get initial state
    self.state = get_data_values(self.tao, self.data_dict)  # Current Data values (current state)
    self.initial_state = self.state
    self.state_dim = len(self.state)
    
    # Get current variable values
    self.var_values = get_variable_values(self.tao, self.var_list) # Current variable values
    self.initial_var_values = self.var_values
    self.action_dim = len(self.var_values)
    # Set up an action space
    self.action_space = spaces.Box(low = action_low * np.ones(self.action_dim),
                                  high = action_high * np.ones(self.action_dim),
                                  dtype = np.float64)
    self.observation_space = spaces.Box(low = obs_low * np.ones(self.state_dim),
                                        high = obs_high * np.ones(self.state_dim),
                                        dtype = np.float64)
    
    # Episode Attributes
    self.done = 0
    self.truncated = 0
    self.convergence_value = convergence_val # Max orbit value is below this = convergence
    self.bad_convergence_value = bad_convergence_val
    self.max_steps = max_steps
    self.nsteps = 0

    # Reward Hyperparameters
    self.alpha = reward_alpha
    self.beta = reward_beta

    # Reward + Convergence type
    self.reward_fun = reward_fun




  ##############################################################################
  # Reset state for next episode
  def reset(self, seed = None, options = None):
    # Reset tao
    self.tao = Tao('-init ' + self.init_file)

    # Reset variable values and current state
    self.state = get_data_values(self.tao, self.data_dict)  # Current Data values (current state)
    self.var_values = get_variable_values(self.tao, self.var_list) # Current variable values

    assert np.sum(self.state == self.initial_state) == len(self.state), "Initialization did not work!"
    assert np.sum(self.var_values == self.initial_var_values) == len(self.var_values), "Initialization did not work!"

    # Reset episode parameters
    self.done = 0
    self.truncated = 0
    self.nsteps = 0

    return self.state, {}
  



  ##############################################################################
  # Sample random actions from uniform distribution
  def sample_action(self):

    # Action values
    action = np.random.uniform(low = self.action_space.low, 
                               high = self.action_space.high, 
                               size=self.action_dim)
    
    return action


    

  ##############################################################################
  # Perform a step (correct variable values and get reward, current_state, and self.done)
  def step(self, action_values : np.ndarray):

    # Update number of steps
    self.nsteps += 1

    # Update model lattice with action
    self.var_values = action_values
    set_new_variable_values(self.tao, self.var_list, self.var_values)

    # Update state
    self.state = get_data_values(self.tao, self.data_dict)

    # Get reward (also checks if episode is done)
    reward = self.calculate_reward()
    
    return self.state, reward, self.done, self.truncated, {}
  




  ##############################################################################
  # Calculate the reward from the given prev and new state
  def calculate_reward(self):

    current_orbit_value = np.max(np.abs(self.state))

    if self.nsteps >= self.max_steps:
      reward = -10
      self.done = 1
      self.truncated = 1
    elif current_orbit_value <= self.convergence_value:
      print(f"convergence achieved, current_orbit_value = {current_orbit_value}")
      self.done = 1
      self.truncated = 0
      reward = 10 + 0.01 * (self.max_steps - self.nsteps)
    elif current_orbit_value >= self.bad_convergence_value:
      #print("bad convergence achieved")
      reward = -10
      self.done = 0
      self.truncated = 0
    else:
      self.done = 0
      self.truncated = 0
      if self.reward_fun == 1:
        reward = self.alpha / current_orbit_value
      elif self.reward_fun == 2:
        reward = 0
      elif self.reward_fun == 3:
        reward = -1 * 1e-2
      elif self.reward_fun == 4:
        reward = 0
        for orbit in self.state:
          if np.abs(orbit) <= self.convergence_value:
            reward += self.alpha
          else:
            reward -= self.beta

    return reward




if __name__ == "__main__":
  env = LTBEnv("./bmad_scripts/tao.init", 
                {"orbit.x": "MW"}, 
                ["correctors_x"],
                -0.01,
                0.01,
                max_steps = 1e3)

  # Store episode
  states = []
  # action_list = []
  # rewards = []
  # dones = []
  # truncateds = []

  # Start of the state
  first_state, _ = env.reset()
  states.append(first_state)

  # Loop through
  start_time = time()
  while True:

    # Get action
    actions = env.sample_action()
    # Make a step
    #new_state, reward, done, truncated, _ = env.step(actions)
    new_state, _, _, done, _ = env.step(actions)

    # Add to storage
    states.append(new_state)
    # rewards.append(reward), dones.append(done), truncateds.append(truncated), action_list.append(actions)
    
    # If done end loop
    if done:
      break

  print(time()-start_time)

  min_max_state = np.inf
  for state in states:
    max_s = np.max(np.abs(state))
    if max_s < min_max_state:
      min_max_state = max_s
    
  resstate, _ = env.reset()

  # print(states)
  # print(rewards)
  # print(resstate)
  print(min_max_state)
  print(time()-start_time)