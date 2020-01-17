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

        #initialize Atoms

        self.num_atoms = 51
        self.v_max = 1
        self.v_min = -1
        self.epsilon = 0.99
        self.delta_z = (self.v_max -self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        self.model = None
        self.target_model = None

        self.memory = deque()
        self.max_memory = 5000
        self.debug_file = open('debug' ,'w')

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
        self.model = self.make_network([1], 51, 2, args.lr)
        self.target_model = self.make_network([1], 51, 2, args.lr)

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
        # Useful functions
        #def make_probas(inputs):
        #    pi = inputs

            # Normalized sigmoid. Gives better results than Softmax
        #    x_exp = K.sigmoid(pi)
        #    return x_exp / K.sum(x_exp)

        #def make_function(input, noutput, activation='sigmoid'):
        #    dense1 = keras.layers.Dense(units=hidden, activation='tanh')(input)
        #    dense2 = keras.layers.Dense(units=noutput, activation=activation)(dense1)

        #    return dense2

        # Neural network with state as input and a probability distribution over
        # actions as output
        #state = keras.layers.Input(shape=(self._actual_state_vars,))

        #pi = make_function(state, self._num_actions, 'linear')                  # Option to execute given current state and option
        #probas = keras.layers.core.Lambda(make_probas, output_shape=(self._num_actions,))(pi)

        #self._model = keras.models.Model(inputs=[state], outputs=[probas])

        # Compile model with Policy Gradient loss
        #print("Compiling model")
        #self._model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mse')
        #print(" done")

        # Policy gradient loss for the policy
        #pi_true = self._model.targets[0]
        #pi_pred = self._model.outputs[0]
        #logpi = K.log(pi_pred + epsilon)
        #grad = K.mean(pi_true * logpi)

        #self._model.total_loss = -grad
        print("Input Shape : " , input_shape , file = self.debug_file)
        #state_input = Input(shape = (1 , ))
        state_input = Input(batch_shape=(1, 1))
        layer1 = Dense(10, activation = 'relu')(state_input)
        layer2 = Dense(10, activation = 'relu')(layer1)
        #flatten = Flatten()(layer2)
        output = Dense(5, activation = 'relu')(layer2)

        distribution_list = []

        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation = 'softmax')(output))

        print("Action Size : " , action_size , file = self.debug_file)
        #print(distribution_list , file = self.debug_file)

        model = Model(input = state_input, output = distribution_list)

        adam = Adam(lr = learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)   

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary,file = self.debug_file) 

        return model


    def encode_state(self, state, timestep, accrued):
        """ Encode a raw state from Gym to a Numpy vector
        """
        if self._discrete_obs:
            # One-hot encode discrete variables
            rs = np.zeros(shape=(self._state_vars,))
            #print("State" , state)
            rs[state] = 1.0
        elif isinstance(state, np.ndarray):
            rs = state.flatten()
        else:
            rs = np.array(state)

        # Add the extra state information
        extratimestep = [(50 - timestep) * 0.1]
        extraaccrued = accrued * 0.1

        if self._extra_state == 'timestep':
            return np.append(rs, extratimestep)
        elif self._extra_state == 'accrued':
            return np.append(rs, extraaccrued)
        elif self._extra_state == 'both':
            return np.append(rs, np.append(extratimestep, extraaccrued))
        else:
            return rs
    
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
        replay_samples = random.sample(self.memory, num_samples)

        state_inputs = np.zeros(((num_samples,)))
        next_states = np.zeros(((num_samples,))) 
        m_prob = [np.zeros((num_samples, self.num_atoms)) for i in range(action_size)]
        action, reward, done = [], [], []

        for i in range(num_samples):
            state_inputs[i] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            #reward.append(replay_samples[i][2])
            next_states[i] = replay_samples[i][3]
            done.append(replay_samples[i][4])

        z = self.model.predict(next_states)
        z_ = self.target_model.predict(next_states)

        cumulative_reward = 0
        
        # Get Optimal Actions for the next states (from distribution z)
        optimal_action_idxs = []
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) # length (num_atoms x num_actions)
        q = q.reshape((num_samples, action_size), order='F')
        optimal_action_idxs = np.argmax(q, axis=1)

        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(num_samples):
            cumulative_reward += reward[i]
            if done[i]: # Terminal State
                # Distribution collapses to a single point

                Tz = min(self.v_max, max(self.v_min, self.scalarize_reward(cumulative_reward)))
                bj = (Tz - self.v_min) / self.delta_z 
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action[i]][i][int(m_l)] += (m_u - bj)
                m_prob[action[i]][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    x = self.scalarize_reward(cumulative_reward) + gamma * self.z[j]
                    #print("X ::" , x, file = self.debug_file)
                    y = max(self.v_min, x)
                    #print("Y ::" , y,  file = self.debug_file)
                    Tz = min(self.v_max, y)
                    #print("TZ ::" , Tz,  file = self.debug_file)
                    bj = (Tz - self.v_min) / self.delta_z 
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxs[i]][i][j] * (m_u - bj)
                    m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxs[i]][i][j] * (bj - m_l)

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
        timestep = 0

        while not done:
            timestep += 1

            # Select an action or option based on the current state
            old_env_state = env_state
            #state = self.encode_state(env_state, timestep, cumulative_rewards)
            #print(state)
            check = random.uniform(0, 1)
            #probas = self.predict_probas(state)
            #action = np.random.choice(self._num_actions, p=probas)
            #print("Encoded State : " , state , file = self.debug_file)
            
            if check < self.epsilon:
                # select random action
                action = round(randint(0, 1))
            else:
                action = self.get_optimal_action(env_state)

            # Store experience, without the reward, that is not yet known
            e = Experience(
                env_state,
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
            e.rewards = rewards

            #print("Reward :: " , rewards,  file = self.debug_file)
            #print("Before" ,  file = self.debug_file)
            self.replay_memory(env_state, action, rewards, env_state, done)
            #print("After" ,  file = self.debug_file)

        # Mark episode boundaries
        self._experiences[-1].interrupt = True


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
            if (i % 32) == 0:
                #learner.learn_from_experiences()
                loss = learner.train_replay()
                print("Episode ::", i,  file= loss_file)
                print("Loss :: " , loss,  file = loss_file)
                learner.memory.clear()
                #learner.update_target_model()

            if (i % 100) == 0:
                print("** Copying Weights **",  file=f)
                learner.update_target_model()

            # decay epsilon
            learner.epsilon = learner.epsilon * 0.99
            if learner.epsilon < 0.001:
                learner.epsilon = 0.001

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
