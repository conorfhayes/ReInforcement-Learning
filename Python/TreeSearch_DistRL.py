from __future__ import print_function
import os
import sys
import argparse

import math
import random
import gym
import numpy as np
from random import randint
import datetime
from collections import deque
from array import *
import collections
import matplotlib.pyplot as plt

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


class Node():

    def __init__(self, env, action, state, id, cumulative_rewards, timestep, layer, file, _dict_, probability, parent,_all_):

        self.simulations = 4
        self.num_actions = 2
        # self.tabular = tabular

        # self.debug_file = open('Node ' + str(id) ,'w')
        self.debug_file = file
        self.children = []
        self.parent = parent
        self.layer = layer + 1
        self.env = env
        self.unique_id = self.layer + 1
        self.num_actions = 2
        self.data = []
        self.name = state
        self.action = action
        self.times_visited = 0
        self.reward = 0
        self.timestep = timestep
        # self._cumulative_rewards_ = cumulative_rewards
        self.cumulative_rewards = cumulative_rewards
        self.id = id
        self.done = False
        self.action = action
        self.probability = probability
        self._all_ = _all_
        
        self.state = state
        self.rewardprobs = {0: {str([0, 0]): 0.45, str([1, 0]): 0.05}, 1: {str([0, 0]): 0.05, str([1, 0]): 0.45}}
        self.rewards = {0: {str([0, 0]): [0, 0], str([1, 0]): [1, 0]}, 1: {str([0, 0]): [0, 0], str([1, 0]): [1, 0]}}
        # self.actionProbs = {0 : {[0,0], [1,0]}, 1: {[0,0], [0,1]}}
        self.rewards_counter = [[[0], [0]], [[0], [0]]]
        self.run(self.state, _dict_, self.probability, self._all_)

    def getDetails(self):

        return self.env, self.action, self.state, self.id, self.cumulative_rewards, self.timestep

    def create_children(self, state, action, cumulative_rewards, _dict_, probability, _all_):
        child_id = self.id + 1

        # env_timestep = child_id + self.timestep
        # for i in range(self.num_actions):.

        self.children.append(
            Node(self.env, action, state, child_id, cumulative_rewards, self.timestep + 1, self.layer, self.debug_file,
                 _dict_, probability, False, _all_))
        self.unique_id += 1

    def getState(self, state, action):

        if state == 0 and action == 1:
            state = 1
        elif state == 1 and action == 1:
            state = 1
        elif state == 0 and action == 0:
            state = 0
        elif state == 0 and action == 0:
            state = 0

        return state

    def simultation(self, state, action, cumulative_rewards, timestep, _dict_):
        next_state, reward, timestep, self.done, __ = self.env.step(action, action, timestep)
        # print("Action :", action, file = self.debug_file)
        # print("Reward :", reward, file = self.debug_file)

        probability = (_dict_[state][action][next_state][str(reward)]['count'] / _dict_[state][action][next_state][
            'count']) / self.num_actions

        self.reward = reward

        return probability, next_state, reward + cumulative_rewards

    def run(self, state, _dict_, probability, _all_):
        timestep = self.timestep
        # state = 0

        if _all_ == True:

            #for action in range(self.num_actions):
            if self.parent == True:
            #    for reward in self.rewardprobs[self.action]:
            #    #if self.parent == True:
                #self.probability = self.rewardprobs[self.action].get(reward)
                #cumulative_rewards = self.cumulative_rewards + self.rewards[self.action].get(str(reward))
                self.probability, next_state, self.cumulative_rewards = self.simultation(self.action, self.action,
                                                                                      self.cumulative_rewards,
                                                                                      timestep, _dict_)

            for action in range(self.num_actions):
                for reward in self.rewardprobs[action]:
                    if self.layer <= self.simulations:
                        cumulative_rewards = self.cumulative_rewards + self.rewards[action].get(str(reward))
                        probability = self.rewardprobs[action].get(reward)

                        if self.probability == 0:
                            self.probability = probability
                        else:
                            self.probability = probability * self.probability

                    if self.layer <= self.simulations and self.done == False:
                        self.create_children(action, action, cumulative_rewards, _dict_, self.probability, _all_)

        else:
            if self.parent == True:
                _probability_, next_state, self.cumulative_rewards = self.simultation(self.action, self.action,
                                                                                      self.cumulative_rewards,
                                                                                      timestep, _dict_)
                state = next_state
                self.probability = _probability_
                # print("Action :", self.action, file = self.debug_file)
                # print("Self Cumulative Rewards :", self.cumulative_rewards, file = self.debug_file)
                # print("Cumulative Rewards :", cumulative_rewards, file = self.debug_file)

            # timestep = self.timestep + 1

            for action in range(self.num_actions):
                if self.layer <= self.simulations:
                    probability, next_state, cumulative_rewards = self.simultation(action, action,
                                                                                   self.cumulative_rewards, timestep,
                                                                                   _dict_)
                    # self.create_children()
                    # print("probability2 : ", probability, file = self.debug_file)

                    if self.probability == 0:
                        self.probability = probability
                    else:
                        self.probability = probability * self.probability
                        self.state = next_state
                        # print("Probability: ", self.probability,file = self.debug_file)

                if self.layer <= self.simulations and self.done == False:
                    # self.cumulative_rewards =  self.simultation(self.action, self.cumulative_rewards, self.timestep)
                    # print("Cumulative Reward : ", self.cumulative_rewards, file = self.debug_file)
                    self.create_children(action, action, cumulative_rewards, _dict_, self.probability, self._all_)

        self.timestep = self.timestep + 1


class Learner(object):
    def __init__(self, args):
        """ Construct a Learner from parsed arguments
        """

        # self.dist_table = [100][8]
        s = 2
        a = 2
        self.num_actions = 2
        self.num_timesteps = 200

        self.dict = {}        

        # Make environment
        self.debug_file = open('debug', 'w')
        self.action_true = [0, 0, 0, 0, 0, 0, 0, 0]
        # self.action_taken = [100][8]

        self.epsilon = 1.0

        self._env = gym.make(args.env)
        self._render = args.render
        self._return_type = args.ret
        self._extra_state = args.extra_state
        self.learning_rate = 0.1
        self.gamma = 1.0

        # Native actions
        aspace = self._env.action_space

        if isinstance(aspace, gym.spaces.Tuple):
            aspace = aspace.spaces
        else:
            aspace = [aspace]  # Ensure that the action space is a list for all the environments

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
            self._state_vars = self._env.observation_space.n  # Prepare for one-hot encoding
        else:
            self._state_vars = np.product(self._env.observation_space.shape)

        if self._extra_state == 'none':
            self._actual_state_vars = self._state_vars
        elif self._extra_state == 'timestep':
            self._actual_state_vars = self._state_vars + 1  # Add the timestep to the state
        elif self._extra_state == 'accrued':
            self._actual_state_vars = self._state_vars + self._num_rewards  # Accrued vector reward
        elif self._extra_state == 'both':
            self._actual_state_vars = self._state_vars + self._num_rewards + 1  # Both addition

        # print('Number of primitive actions:', self._num_actions)
        # print('Number of state variables', self._actual_state_vars)
        # print('Number of objectives', self._num_rewards)

        # Lists for policy gradient
        self._experiences = []

    def getResults(self, tree):

        dist_rewards = []
        dist_prob = []

        a = 0

        # for node in tree:
        # for node in range(2):
        def _getResults(tree):
            if len(tree.children) == 0:
                dist_rewards.append(tree.cumulative_rewards)
                dist_prob.append(tree.probability)
                # print("Dist Rewards : " ,dist_rewards, file = self.debug_file)

            for n in range(len(tree.children)):
                _getResults(tree.children[n])

        _getResults(tree)

        # if len(tree.children) == 0:
        #    dist_reward.append(tree.children.cumulative_rewards)

        # print("Dist Rewards : " ,dist_rewards, file = self.debug_file)
        return dist_rewards, dist_prob

    def rollOut(self, env, action, state, id, cumulative_rewards, timestep, unique_id, file, _dict_, parent, _all_):

        # self.rollOut(self._env, action,env_state, 0, cumulative_rewards, self.timestep, 0)

        tree = Node(env, action, state, id, cumulative_rewards, timestep, unique_id, file, _dict_, 0, parent, _all_)
        action_rewards, action_prob = self.getResults(tree)
        """
        print("CR", cumulative_rewards, file = file)
        print("Layer A0 1 State ", tree.state, file = file)
        print("Layer A0 1 R ", tree.reward, file = file)
        print("Layer A0 2 State ", tree.children[0].state, file = file)
        print("Layer A0 2 R ", tree.children[0].reward, file = file)
        print("Layer A0 3 State ", tree.children[0].children[0].state, file = file)
        print("Layer A0 3 R ", tree.children[0].children[0].reward, file = file)
        print("Layer A0 4 State ", tree.children[0].children[0].children[0].state, file = file)
        print("Layer A0 4 R ", tree.children[0].children[0].children[0].reward, file = file)
        print("Layer A0 5 State", tree.children[0].children[0].children[0].children[0].state, file = file)
        print("Layer A0 5 R", tree.children[0].children[0].children[0].children[0].reward, file = file)

        print("CR", cumulative_rewards, file = file)
        print("Layer A0 1 CR ", tree.cumulative_rewards, file = file)
        print("Layer A0 2 CR ", tree.children[0].cumulative_rewards, file = file)
        print("Layer A0 3 CR ", tree.children[0].children[0].cumulative_rewards, file = file)
        print("Layer A0 4 CR ", tree.children[0].children[0].children[0].cumulative_rewards, file = file)
        print("Layer A0 5 CR", tree.children[0].children[0].children[0].children[0].cumulative_rewards, file = file)

        print("CR", cumulative_rewards, file = file)
        print("Layer A1 1 R ", tree.reward, file = file)
        print("Layer A1 2 R ", tree.children[1].reward, file = file)
        print("Layer A1 3 R ", tree.children[1].children[1].reward, file = file)
        print("Layer A1 4 R ", tree.children[1].children[1].children[1].reward, file = file)
        print("Layer A1 5 R", tree.children[1].children[1].children[1].children[1].reward, file = file)

        print("CR", cumulative_rewards, file = file)
        print("Layer A1 1 CR ", tree.cumulative_rewards, file = file)
        print("Layer A1 2 CR ", tree.children[1].cumulative_rewards, file = file)
        print("Layer A1 3 CR ", tree.children[1].children[1].cumulative_rewards, file = file)
        print("Layer A1 4 CR ", tree.children[1].children[1].children[1].cumulative_rewards, file = file)
        print("Layer A1 5 CR", tree.children[1].children[1].children[1].children[1].cumulative_rewards, file = file)
        """

        return action_rewards, action_prob

    def encode_state(self, state, timestep, accrued, reward, previous_reward):
        """ Encode a raw state from Gym to a Numpy vector
        """
        index = 0
        if self._discrete_obs:
            # One-hot encode discrete variables
            rs = np.zeros(shape=(self._state_vars,))
            # print("State" , state)
            rs[state] = 1.0
        elif isinstance(state, np.ndarray):
            rs = state.flatten()
        else:
            rs = np.array(state)

        # Add the extra state information
        extratimestep = [(50 - timestep) * 0.1]
        extraaccrued = accrued * 0.1

        index = np.where(rs == 1)
        index = index[0][0]

        return index

    def encode_reward(self, reward):
        """ Encode a scalar or vector reward as an array
        """
        if ifinstance(reward, float):
            return np.array([reward])
        else:
            return np.array(reward)   

    def scalarize_reward(self, rewards):
        """ Return a scalarized reward from objective scores
        """
        if self._utility is None:
            # Default scalarization, just a sum
            return np.sum(rewards)
        else:
            # Use the user utility function
            return eval(self._utility, {}, {'r' + str(i + 1): rewards[i] for i in range(self._num_rewards)})

    

    def getNextState(self, state, action):

        if action == 0:
            return 0

        if action == 1:
            return 1
    

    def updateDictionary(self, env_state, action, new_env_state, rewards, _dict_):

        if env_state in _dict_:
            if action in _dict_[env_state]:
                if new_env_state in _dict_[env_state][action]:
                    if str(rewards) in _dict_[env_state][action][new_env_state]:
                        pass
                    else:
                        _dict_[env_state][action][new_env_state].update({str(rewards): {'count': 0}})
                else:
                    _dict_[env_state][action].update({new_env_state: {str(rewards): {'count': 0}}, 'count': 0})
            else:
                _dict_[env_state].update({action: {new_env_state: {str(rewards): {'count': 0}, 'count': 0}}})
        else:
            _dict_.update({env_state: {action: {new_env_state: {str(rewards): {'count': 0}, 'count': 0}}}})

        if str(rewards) in _dict_[env_state][action][new_env_state]:
            pass
        else:
            _dict_[env_state][action][new_env_state][str(rewards)]['count'] = 0

        return _dict_

    def graph(self, _dict_):

        plt.bar(range(len(_dict_)), list(_dict_.values()), align='center', alpha=0.5)
        plt.xticks(range(len(_dict_)), list(_dict_.keys()))
        # matplotlib.pyplot.show()
        plt.show(block=True)

        return

    def run(self):
        """ Execute an option on the environment
        """
        env_state = self._env.reset()

        done = False

        cumulative_rewards = np.zeros(shape=(self._num_rewards,))
        rewards = np.zeros(shape=(self._num_rewards,))
        previous_culmulative_rewards = np.zeros(shape=(self._num_rewards,))

        timestep = 0
        scalar_reward = 0

        action = -10
        # action_ = 0
        time_since_utility = 0

        # self.dict = lambda : defaultdict(self.dict)

        action_taken = np.zeros(shape=(self._num_rewards,))
        times_action_taken = np.zeros(shape=(self._num_rewards,))

        reward_wood = np.zeros(shape=(self._num_rewards,))
        reward_fish = np.zeros(shape=(self._num_rewards,))
        previous_utility = 0
        self.timestep = 0

        previous_reward = [0, 0]
        previous_action = 0
        new_env_state = -1

        rewards_ = []
        # rewards_recieved = [[0 for x in range(2)] for y in range(1)]
        

        # potential_reward_action.append([])
        # potential_reward_action.append([])

        while not done:
            self.timestep += 1
            # print(" *** timestep *** : " , timestep , file = self.debug_file)
            # Select an action or option based on the current state

            state = self.encode_state(env_state, timestep, cumulative_rewards, cumulative_rewards,
                                      previous_culmulative_rewards)
            state = env_state
            # print("State :: " , state,  file = self.debug_file)
            # print(state)
            check = random.uniform(0, 1)

            increase_utility = 0

            potential_reward_action = []
            potential_reward_action.append([])
            potential_reward_action.append([])
            prob_reward = 0

            if check < self.epsilon:
                # select random action
                action = random.randint(0, 1)
            #    print("Random Action : " , action , file = self.debug_file)
            else:

                best_prob = -2147483648
                best_prob = 0
                actionTaken = 0
                rewardTaken = 0
                times_action_taken = 0
                reward_prob = 0
                next_state = 0
                rewards = []
                valueCheck = 0
                actionPicked = False
                action = 0
                action_rewards = []
                action_prob = []
                dictResults = {}
                _all_ = True

                for i in range(self.num_actions):
                    dictResults.update({i: {}})

                for _action_ in range(self.num_actions):

                    # env, action, state, id, cumulative_rewards, timestep)

                    action_rewards.append(
                        self.rollOut(self._env, _action_, _action_, 0, cumulative_rewards, self.timestep, 0,
                                     self.debug_file, self.dict, True, _all_)[0])
                    action_prob.append(
                        self.rollOut(self._env, _action_, _action_, 0, cumulative_rewards, self.timestep, 0,
                                     self.debug_file, self.dict, True, _all_)[1]);

                    for i in range(len(action_rewards[_action_])):

                        if str(action_rewards[_action_][i]) in dictResults[_action_]:
                            
                            d1 = {str(action_rewards[_action_][i]): dictResults[_action_].get(
                                str(action_rewards[_action_][i])) + action_prob[_action_][i]}
                            
                            dictResults[_action_].update(d1)

                        else:
                            dictResults[_action_].update({str(action_rewards[_action_][i]): action_prob[_action_][i]})

                
                print('Cumulative Rewards', cumulative_rewards, file=self.debug_file)
                print("Dictionary Action 0 :: ", dictResults[0], file=self.debug_file)
                print("Dictionary Action 1 :: ", dictResults[1], file=self.debug_file)
                print("Action Rewards :: ", action_rewards, file=self.debug_file)

                #self.graph(dictResults[0])
            
            
            if len(self._aspace) > 1:
                # Choose each of the factored action depending on the composite action
                actions = [0] * len(self._aspace)
                a = action

                for i in range(len(actions)):
                    actions[i] = a % self._aspace[i].n
                    a //= self._aspace[i].n

                new_env_state, rewards, self.timestep, done, __ = self._env.step(actions)
            else:
                # Simple scalar action
                new_env_state, rewards, self.timestep, done, __ = self._env.step(env_state, action, self.timestep)

            if self._render:
                self._env.render()

            cumulative_rewards += rewards
            
            self.dict = self.updateDictionary(env_state, action, new_env_state, rewards, self.dict)
            

            self.dict[env_state][action][new_env_state]['count'] += 1
            self.dict[env_state][action][new_env_state][str(rewards)]['count'] += 1

            
            env_state = new_env_state

            # Mark episode boundaries

        # print("Prob", self.prob_time_table[env_state][timestep][:], file = self.debug_file)
        # previous_culmulative_rewards = cumulative_rewards
        return cumulative_rewards


def main():
    # Parse parameters
    parser = argparse.ArgumentParser(description="Reinforcement Learning for the Gym")

    parser.add_argument("--render", action="store_true", default=False,
                        help="Enable a graphical rendering of the environment")
    parser.add_argument("--monitor", action="store_true", default=False, help="Enable Gym monitoring for this run")
    parser.add_argument("--env", required=True, type=str, help="Gym environment to use")
    parser.add_argument("--avg", type=int, default=1, help="Episodes run between gradient updates")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")
    parser.add_argument("--name", type=str, default='', help="Experiment name")

    parser.add_argument("--ret", type=str, choices=['forward', 'both'], default='both',
                        help='Type of return used for training, only forward-looking or also using accumulated rewards')
    parser.add_argument("--utility", type=str, help="Utility function, a function of r1 to rN")
    parser.add_argument("--extra-state", type=str, choices=['none', 'timestep', 'accrued', 'both'], default='none',
                        help='Additional information given to the agent, like the accrued reward')
    parser.add_argument("--hidden", default=50, type=int, help="Hidden neurons of the policy network")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate of the neural network")

    # Next and Sub from arguments
    args = parser.parse_args()

    # Instantiate learner
    learner = Learner(args)
    # learner.model = self.make_network()

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
            # print("Episode ::", i,  file=f)
            if i == 0:
                avg = rewards
            else:
                avg = 0.99 * avg + 0.01 * rewards

                # decay epsilon
            learner.epsilon = learner.epsilon * 0.9
            if learner.epsilon < 0.001:
                learner.epsilon = 0.001

            scalarized_avg = learner.scalarize_reward(avg)

            print("Cumulative reward:", rewards, "; average rewards:", avg, scalarized_avg, file=f)
            # print(args.name, "Cumulative reward:", rewards, "; average rewards:", avg, file = f )
            # print(scalarized_avg, file = f)
            f.flush()

    except KeyboardInterrupt:
        pass

    if args.monitor:
        learner._env.monitor.close()

    f.close()


if __name__ == '__main__':
    main()