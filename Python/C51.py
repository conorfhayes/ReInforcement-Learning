from __future__ import print_function
import os
import sys
import argparse

import keras
import math
import random
import keras.backend as K
import gym
import numpy as np
from random import randint
import datetime
import tensorflow as tf
from collections import deque
from keras.layers import Dense, Input, Activation
from keras.models import Sequential, load_model, Model
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.core import Dense, Dropout, Activation, Flatten
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

from gym.envs.registration import register



register(
    id='RandomMOMDP-v0',
    entry_point='randommomdp:RandomMOMDP',
    reward_threshold=0.0,
    kwargs={'nstates': 100, 'nobjectives': 4, 'nactions': 8, 'nsuccessor': 12, 'seed': 1}
)
register(
    id='FishWood-v0',
    entry_point='fishwood:FishWood',
    reward_threshold=0.0,
    kwargs={'fishproba': 0.1, 'woodproba': 0.9}
)

class Experience(object):
    def __init__(self, state, action):
        """ In <state>, <action> has been choosen, which led to <reward>
        """
        self.state = state
        self.action = action
        self.rewards = None             # The rewards are set once they are known
        self.interrupt = False         # Set to true at the end of an episode

class Learner(object):
    def __init__(self, args):
        """ Construct a Learner from parsed arguments
        """

        # Make environment
        self._env = gym.make(args.env)
        self._render = args.render
        self._return_type = args.ret
        self._extra_state = args.extra_state
        self.learning_rate = 0.00001
        self.gamma = 1.0
        #initialize Atoms

        self.num_atoms = 51
        self.v_max = 1
        self.v_min = -1
        self.epsilon = 0.99
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        self.model = None
        self.target_model = None

        self.memory = deque()
        self.max_memory = 5000
        self.debug_file = open('debug' ,'w')

        self.action_wood = 0
        self.action_fish = 0
        self.action_wood_true = 0
        self.action_fish_true = 0
        self.prob_wood = 0
        self.prob_fish = 0

        self.time_to_train = False

        # Native actions
        aspace = self._env.action_space

        if isinstance(aspace, gym.spaces.Tuple):
            aspace = aspace.spaces
        else:
            aspace = [aspace]               # Ensure that the action space is a list for all the environments

        self._num_rewards = getattr(self._env, 'num_rewards', 1)
        self._num_actions = np.prod([a.n for a in aspace])
        self._aspace = aspace

        # Make an utility function
        if args.utility is not None:
            self._utility = compile(args.utility, 'utility', 'eval')
        else:
            self._utility = None

        # Build network
        self._discrete_obs = isinstance(self._env.observation_space, gym.spaces.Discrete)

        if self._discrete_obs:
            self._state_vars = self._env.observation_space.n                    # Prepare for one-hot encoding
        else:
            self._state_vars = np.product(self._env.observation_space.shape)

        if self._extra_state == 'none':
            self._actual_state_vars = self._state_vars
        elif self._extra_state == 'timestep':
            self._actual_state_vars = self._state_vars + 1                      # Add the timestep to the state
        elif self._extra_state == 'accrued':
            self._actual_state_vars = self._state_vars + self._num_rewards      # Accrued vector reward
        elif self._extra_state == 'both':
            self._actual_state_vars = self._state_vars + self._num_rewards + 1  # Both addition

        #self.make_network(args.hidden, args.lr)
        self.model = self.make_network([3], self.num_atoms, 2, self.learning_rate)
        self.target_model = self.make_network([3], self.num_atoms, 2, self.learning_rate)

        print('Number of primitive actions:', self._num_actions)
        print('Number of state variables', self._actual_state_vars)
        print('Number of objectives', self._num_rewards)

        # Lists for policy gradient
        self._experiences = []

    def update_target_model(self):

        self.target_model.set_weights(self.model.get_weights())

    def make_network(self, input_shape, num_atoms, action_size, learning_rate):
        """ Initialize a simple multi-layer perceptron for policy gradient
        """
        
        print("Input Shape : " , input_shape , file = self.debug_file)

        state_input = Input(shape = (3 , ))
        layer1 = Dense(100, activation = 'relu')(state_input)    
        #output = Dense(5, activation = 'relu')(layer1)

        distribution_list = []

        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation = 'softmax')(layer1))

        print("Action Size : " , action_size , file = self.debug_file)
        #print(distribution_list , file = self.debug_file)

        model = Model(input = state_input, output = distribution_list)

        adam = Adam(lr = learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)   

        return model


    def encode_state(self, state, timestep, accrued, reward, previous_reward):
        """ Encode a raw state from Gym to a Numpy vector
        """
        if self._discrete_obs:
            # One-hot encode discrete variables
            rs = np.zeros(shape=(2,))
            #print("State" , state)
            rs[state] = 1.0
        elif isinstance(state, np.ndarray):
            rs = state.flatten()
        else:
            rs = np.array(state)

        # Add the extra state information
        extratimestep = [(50 - timestep) * 0.1]
        extraaccrued = accrued * 0.1

        
        
        #np.append(rs, previous_reward)

        #np.append(rs, reward)
        #np.append(rs, previous_reward)

        
            #np.append(rs, reward)
            #np.append(rs, previous_reward)
        return np.append(rs, timestep)
            
    
    def encode_reward(self, reward):
        """ Encode a scalar or vector reward as an array
        """
        if ifinstance(reward, float):
            return np.array([reward])
        else:
            return np.array(reward)

    def predict_probas(self, state):
        """ Return a probability distribution over actions given the current state.
        """
        output = self._model.predict_on_batch([
            state.reshape((1, self._actual_state_vars))
        ])

        return output[0].flatten()

    def scalarize_reward(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        if self._utility is None:
            # Default scalarization, just a sum
            return np.sum(rewards)
        else:
            # Use the user utility function
            return eval(self._utility, {}, {'r'+str(i+1): rewards[i] for i in range(self._num_rewards)})

    def learn_from_experiences(self):
        """ Learn from the experience pool, using Policy Gradient
        """
        N = len(self._experiences)

        if N == 0:
            return

        target_action = np.zeros(shape=(N, self._num_actions))
        source_state = np.zeros(shape=(N, self._actual_state_vars))

        # Compute forward-looking cumulative rewards
        forward_cumulative_rewards = np.zeros(shape=(N, self._num_rewards))
        backward_cumulative_rewards = np.zeros(shape=(N, self._num_rewards))
        cumulative_reward = np.zeros(shape=(1, self._num_rewards))

        for i in range(N-1, -1, -1):
            e = self._experiences[i]

            if e.interrupt:
                cumulative_reward.fill(0.0)     # Break the cumulative reward chain

            cumulative_reward += e.rewards
            forward_cumulative_rewards[i] = cumulative_reward

        # Compute the backward-looking cumulative reward
        cumulative_reward.fill(0.0)

        for i in range(N):
            e = self._experiences[i]

            cumulative_reward += e.rewards
            backward_cumulative_rewards[i] = cumulative_reward

            if e.interrupt:
                cumulative_reward.fill(0.0)

        # Build source and target arrays for the actor
        for i in range(N):
            e = self._experiences[i]

            # Scalarize the return
            value = self.scalarize_reward(forward_cumulative_rewards[i])

            if self._return_type == 'both':
                value += self.scalarize_reward(backward_cumulative_rewards[i])

            target_action[i, e.action] = value
            source_state[i, :] = e.state

        # Train the neural network
        self._model.fit(
            [source_state],
            [target_action],
            batch_size=N,
            epochs=1,
            verbose=0
        )

        # Prepare for next episode
        self._experiences.clear()

    def replay_memory(self, s_t, action_idx, r_t, s_t1, is_terminated):

        self.memory.append((s_t, action_idx, r_t, s_t1, is_terminated))        

    def train_replay(self):
        print("In replay_memory : " , file = self.debug_file)
        num_samples = 32
        state_size = 2
        action_size = 2
        num_atoms = 51
        gamma = 0.99
        replay_samples = []

        #for i in range(num_samples):
        #    replay_samples[i] = self.memory[i]

        replay_samples = random.sample(self.memory, num_samples)
        #replay_samples = self.memory

        state_inputs = np.zeros(shape = (num_samples, 3))
        next_states = np.zeros(shape = (num_samples, 3))
        m_prob = [np.zeros((num_samples, self.num_atoms)) for i in range(action_size)]
        action, reward, done = np.zeros(shape = (num_samples, 1)), np.zeros(shape = (num_samples, 1)), np.zeros(shape = (num_samples, 1))

        #for i in range(num_samples):

        #print("** State **",replay_samples[0][0] , file = self.debug_file)
        #print("** Action **",replay_samples[0][1] , file = self.debug_file)
        #print("** Action **",replay_samples[0], file = self.debug_file)
        #print("** Reward **",replay_samples[0][2] , file = self.debug_file)
        #print("** Next State **",replay_samples[0][3] , file = self.debug_file)
        #print("** Done **",replay_samples[0][4] , file = self.debug_file)

        for i in range(num_samples):
            
            #print("** Memory **",replay_samples[i] , file = self.debug_file)
            state_inputs[i] = replay_samples[i][0]
            action[i] = replay_samples[i][1]
            reward[i] = replay_samples[i][2]
            next_states[i] = replay_samples[i][3]
            done[i] = replay_samples[i][4]
        
            
        z = self.model.predict(np.array(next_states))
        z_ = self.target_model.predict(np.array(next_states))

        cumulative_reward = np.zeros(shape=(1, self._num_rewards))
        
        # Get Optimal Actions for the next states (from distribution z)
        optimal_action_idxs = []
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) # length (num_atoms x num_actions)
        q = q.reshape((num_samples, action_size), order='F')
        optimal_action_idxs = np.argmax(q, axis=1)

        forward_cumulative_rewards = np.zeros(shape=(num_samples, 2))

        for i in range(num_samples - 1 , -1, -1):
            #e = self._experiences[i]

            #if e.interrupt:
            #    cumulative_reward.fill(0.0)     # Break the cumulative reward chain
            #print("***************", i , file = self.debug_file)
            cumulative_reward += reward[i]
            forward_cumulative_rewards[i] = cumulative_reward

        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(num_samples):

            #cumulative_reward += reward[i]

            if done[i]: # Terminal State
                # Distribution collapses to a single point
                #print("** reward i **",reward[i] , file = self.debug_file)
                Tz = min(self.v_max, max(self.v_min, reward[i]))
                bj = (Tz - self.v_min) / self.delta_z 
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[int(action[i])][i][int(m_l)] += (m_u - bj)
                m_prob[int(action[i])][i][int(m_u)] += (bj - m_l)

            else:
                for j in range(self.num_atoms):
                    #print("** reward i **",reward[i] , file = self.debug_file)
                    Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))                    
                    bj = (Tz - self.v_min) / self.delta_z 
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    
                    m_prob[int(action[i])][i][int(m_l)] += z_[optimal_action_idxs[i]][i][j] * (m_u - bj)
                    m_prob[int(action[i])][i][int(m_u)] += z_[optimal_action_idxs[i]][i][j] * (bj - m_l)

        loss = self.model.fit(state_inputs, m_prob, batch_size = num_samples, nb_epoch=1, verbose=0)

        #self.memory.clear()
        return loss.history['loss']


    def get_optimal_action(self, state):
        print("State : " , state , file = self.debug_file)
       
        state_array = np.array([state])
        print("State Array : " , state_array , file = self.debug_file)
        z = self.model.predict(state_array)      
        print(len(z), file = self.debug_file) 
        print(z , file = self.debug_file)
        self.debug_file.flush()
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) 
        print(q , file = self.debug_file)

        # Pick action with the biggest Q value
        action_idx = np.argmax(q)
        print(action_idx , file = self.debug_file)
        
        return action_idx

    def run(self):
        """ Execute an option on the environment
        """
        env_state = self._env.reset()

        done = False

        cumulative_rewards = np.zeros(shape=(self._num_rewards,))
        rewards = np.zeros(shape=(self._num_rewards,))
        previous_culmulative_rewards = np.zeros(shape=(self._num_rewards,))

        potential_reward_fish = np.zeros(shape=(self._num_rewards,))
        potential_reward_wood = np.zeros(shape=(self._num_rewards,))

        potential_reward_fish1 = np.zeros(shape=(self._num_rewards,))
        potential_reward_wood1 = np.zeros(shape=(self._num_rewards,))

        potential_reward_fish2 = np.zeros(shape=(self._num_rewards,))
        potential_reward_wood2 = np.zeros(shape=(self._num_rewards,))

        potential_reward_fish3 = np.zeros(shape=(self._num_rewards,))
        potential_reward_wood3 = np.zeros(shape=(self._num_rewards,))

        timestep = 0
        scalar_reward = 0
        

        action = 0



        while not done:
            timestep += 1
            print(" *** timestep *** : " , timestep , file = self.debug_file)
            # Select an action or option based on the current state            
            old_env_state = env_state

            state = self.encode_state(env_state, timestep, cumulative_rewards, cumulative_rewards, previous_culmulative_rewards)
            print("State :: " , state,  file = self.debug_file)
            #print(state)

            #probas = self.predict_probas(state)
            #action = np.random.choice(self._num_actions, p=probas)
            #print("Encoded State : " , state , file = self.debug_file)
            
                
            if timestep > 32:
                self.time_to_train = True

            if self.time_to_train == True:
                loss = self.train_replay()

            #potential_reward_fish = [1,0] + cumulative_rewards
            
            #potential_reward_wood = [0,1] + cumulative_rewards 

            #if self.scalarize_reward(potential_reward_fish) - self.scalarize_reward(cumulative_rewards) == 1:
                
            #    action = 0

            #elif self.scalarize_reward(potential_reward_wood) - self.scalarize_reward(cumulative_rewards) == 1:
                
            #    action = 1

            #else:
            #    action = round(randint(0, 1))
            wood = []
            fish = []
            action_holder = []


            self.reward_value_wood = rewards[1]
            self.reward_value_fish = rewards[0]

            self.action_wood_true += self.reward_value_wood
            self.action_fish_true += self.reward_value_fish

            if timestep > 1:
                self.prob_fish = self.action_fish_true / self.action_fish
                self.prob_wood = self.action_wood_true / self.action_wood

                print("action_fish_true : " , self.action_fish_true , file = self.debug_file)
                print("action_fish : " , self.action_fish , file = self.debug_file)
            

            potential_reward_wood = cumulative_rewards
            potential_reward_wood1 = (self.scalarize_reward(([0,0] + cumulative_rewards)) - self.scalarize_reward(cumulative_rewards)) * self.prob_fish
            potential_reward_wood2 = (self.scalarize_reward(([0,1] + cumulative_rewards)) - self.scalarize_reward(cumulative_rewards)) * self.prob_wood
            potential_reward_wood3 = (self.scalarize_reward(([1,0] + cumulative_rewards)) - self.scalarize_reward(cumulative_rewards)) * 0

            potential_reward_fish = cumulative_rewards
            potential_reward_fish1 = (self.scalarize_reward(([0,0] + cumulative_rewards)) - self.scalarize_reward(cumulative_rewards)) * self.prob_wood
            potential_reward_fish2 = (self.scalarize_reward(([0,1] + cumulative_rewards)) - self.scalarize_reward(cumulative_rewards)) * 0
            potential_reward_fish3 = (self.scalarize_reward(([1,0] + cumulative_rewards)) - self.scalarize_reward(cumulative_rewards)) * self.prob_fish

            #print("potential_reward_fish3 : " , ([1,0] + cumulative_rewards) , file = self.debug_file)
            #print("potential_reward_fish3 ns : " , self.scalarize_reward(([1,0] + cumulative_rewards)) , file = self.debug_file)
            #print("potential_reward_fish3 ns2 : " , self.scalarize_reward(([1,0] + cumulative_rewards)) - self.scalarize_reward(cumulative_rewards)  , file = self.debug_file)

            print("Prob Fish : " , self.prob_fish , file = self.debug_file)
            print("Prob Wood : " , self.prob_wood , file = self.debug_file)

            wood.append(potential_reward_wood1)
            wood.append(potential_reward_wood2)
            wood.append(potential_reward_wood3)

            wood = [potential_reward_wood1, potential_reward_wood2, potential_reward_wood3]
            fish = [potential_reward_fish1, potential_reward_fish2, potential_reward_fish3]           

            #print("wood : " , wood , file = self.debug_file)
            #print("fish : " , fish , file = self.debug_file)

            max_wood = max(wood)
            max_fish = max(fish)

            #print("Max Wood : " , max_wood , file = self.debug_file)
            #print("Max Fish : " , max_fish , file = self.debug_file)
            
            

            if max_fish == 0.0 and max_wood == 0.0:
                #print("Max : " , max_fish, max_wood , file = self.debug_file)
                max_fish = self.prob_fish
                max_wood = self.prob_wood

            action_holder = [max_fish, max_wood]

            
            check = random.uniform(0, 1)

            if check < self.epsilon:
                # select random action
                action = random.randint(0, 1)
            #    print("Random Action : " , action , file = self.debug_file)
            else:
                #action = action_holder.index(max(action_holder))
                action = self.get_optimal_action(state)

            if action == 0:
                self.action_fish += 1

            if action == 1:
                self.action_wood += 1
            

            # Store experience, without the reward, that is not yet known
            e = Experience(
                state,
                action
            )
            self._experiences.append(e)

            # Execute the action
            if len(self._aspace) > 1:
                # Choose each of the factored action depending on the composite action
                actions = [0] * len(self._aspace)
                a = action

                for i in range(len(actions)):
                    actions[i] = a % self._aspace[i].n
                    a //= self._aspace[i].n

                env_state, rewards, done, __ = self._env.step(actions)
            else:
                # Simple scalar action
                env_state, rewards, done, __ = self._env.step(action)

            if self._render:
                self._env.render()

            # Update the experience with its reward
            
            cumulative_rewards += rewards
            previous_culmulative_rewards = cumulative_rewards - rewards

            reward = self.scalarize_reward(cumulative_rewards) - self.scalarize_reward(previous_culmulative_rewards)
            min_reward = [0,0]
            compare = np.subtract(cumulative_rewards, previous_culmulative_rewards) 

            if np.array_equal(compare, min_reward) == 0:
                if reward == 0.0:
                    scalar_reward = -0.1
                elif reward == 1.0:
                    scalar_reward = 1
            else:
                scalar_reward = 0.1

            #scalar_reward = reward
            
            print(" **** Scalar Reward **** : " , scalar_reward , file = self.debug_file)
            print(" **** Previous Culmulative Rewards **** : " , previous_culmulative_rewards , file = self.debug_file)
            print(" **** Culmulative Rewards **** : " , cumulative_rewards , file = self.debug_file)
            print(" **** Reward **** : " , reward , file = self.debug_file)
            print(" **** Compare **** : " , compare , file = self.debug_file)


            e.rewards = rewards

            next_timestep = 0

            if timestep == 200:
                next_timestep = 1
            else:
                next_timestep = timestep + 1

            next_state = self.encode_state(env_state, next_timestep, cumulative_rewards, cumulative_rewards, previous_culmulative_rewards)
            
            #print("Next State :: " , next_state,  file = self.debug_file)
            #print("Before" ,  file = self.debug_file)
            self.replay_memory(state, action, scalar_reward, next_state, done)
            #print("After" ,  file = self.debug_file)

        # Mark episode boundaries
        self._experiences[-1].interrupt = True

        #previous_culmulative_rewards = cumulative_rewards
        return cumulative_rewards



def main():
    # Parse parameters
    parser = argparse.ArgumentParser(description="Reinforcement Learning for the Gym")

    parser.add_argument("--render", action="store_true", default=False, help="Enable a graphical rendering of the environment")
    parser.add_argument("--monitor", action="store_true", default=False, help="Enable Gym monitoring for this run")
    parser.add_argument("--env", required=True, type=str, help="Gym environment to use")
    parser.add_argument("--avg", type=int, default=1, help="Episodes run between gradient updates")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")
    parser.add_argument("--name", type=str, default='', help="Experiment name")

    parser.add_argument("--ret", type=str, choices=['forward', 'both'], default='both', help='Type of return used for training, only forward-looking or also using accumulated rewards')
    parser.add_argument("--utility", type=str, help="Utility function, a function of r1 to rN")
    parser.add_argument("--extra-state", type=str, choices=['none', 'timestep', 'accrued', 'both'], default='none', help='Additional information given to the agent, like the accrued reward')
    parser.add_argument("--hidden", default=50, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate of the neural network")

    # Next and Sub from arguments
    args = parser.parse_args()

    # Instantiate learner
    learner = Learner(args)
    #learner.model = self.make_network()

    # Learn
    f = open('out-' + args.name, 'w')
    loss_file = open('loss_file', 'w')

    if args.monitor:
        learner._env.monitor.start('/tmp/monitor', force=True)

    try:
        old_dt = datetime.datetime.now()
        avg = np.zeros(shape=(learner._num_rewards,))

        for i in range(args.episodes):
            rewards = learner.run()
            print("Episode ::", i,  file=f)
            if i == 0:
                avg = rewards
            else:
                avg = 0.99 * avg + 0.01 * rewards

            # Learn when enough experience is accumulated
            #if (i % args.avg) == 0:

            #if (i % 10) == 0 and i < 20:
            #    learner.memory.clear()

            #if  i > 1:
                #learner.learn_from_experiences()
            #    loss = learner.train_replay()
            #    print("Episode ::", i,  file= loss_file)
            #    print("Loss :: " , loss,  file = loss_file)
                #learner.update_target_model()


            if (i % 1.5) == 0:
                print("** Copying Weights **",  file=f)
                learner.update_target_model()

            # decay epsilon
            learner.epsilon = learner.epsilon * 0.99
            if learner.epsilon < 0.1:
                learner.epsilon = 0.1

            scalarized_avg = learner.scalarize_reward(avg)

            print("Cumulative reward:", rewards, "; average rewards:", avg, scalarized_avg, file=f)
            print(args.name, "Cumulative reward:", rewards, "; average rewards:", avg, scalarized_avg)
            print("Epsilon ::", learner.epsilon, file = f)
            f.flush()

    except KeyboardInterrupt:
        pass

    if args.monitor:
        learner._env.monitor.close()

    f.close()

if __name__ == '__main__':
    main()
