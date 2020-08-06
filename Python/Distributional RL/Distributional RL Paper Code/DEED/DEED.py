import gym
import numpy as np

class DEED(gym.Env):

    def __init__(self):
        super(DEED, self).__init__()

        self.u_holder = np.array([
            [150, 470, 786.7988, 38.5397, 0.1524, 450, 0.041, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80],
            [135, 470, 451.3251, 46.1591, 0.1058, 600, 0.036, 103.3908, -2.4444, 0.0312, 0.5035, 0.0207, 80, 80],
            [73, 340, 1049.9977, 40.3965, 0.0280, 320, 0.038, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 80, 80],
            [60, 300, 1243.5311, 38.3055, 0.0354, 260, 0.052, 300.3910, -4.0695, 0.0509, 0.4968, 0.0202, 50, 50],
            [73, 243, 1658.5696, 36.3278, 0.0211, 280, 0.063, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50],
            [57, 160, 1356.6592, 38.2704, 0.0179, 310, 0.048, 320.0006, -3.8132, 0.0344, 0.4972, 0.0200, 50, 50],
            [20, 130, 1450.7045, 36.5104, 0.0121, 300, 0.086, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30],
            [47, 120, 1450.7045, 36.5104, 0.0121, 340, 0.082, 330.0056, -3.9023, 0.0465, 0.5163, 0.0214, 30, 30],
            [20, 80, 1455.6056, 39.5804, 0.1090, 270, 0.098, 350.0056, -3.9524, 0.0465, 0.5475, 0.0234, 30, 30],
            [10, 55, 1469.4026, 40.5407, 0.1295, 380, 0.094, 360.0012, -3.9864, 0.0470, 0.5475, 0.0234, 30, 30],
        ])

        # Optimal power output for each generator at each timestep
        self.generators = np.array([
            [137,179,272,288,230,264,282,308,383,352,395,409,358,304,242,165,150,209,282,329,286,161,127,157],
            [274,245,245,245,251,314,291,294,337,380,407,430,384,354,314,241,218,261,337,414,344,341,264,211],
            [126,134,136,150,176,197,208,247,250,295,314,324,335,258,200,171,171,221,226,274,327,282,210,171],
            [160,115,124,165,174,179,214,219,250,295,298,298,298,298,298,293,276,267,250,269,269,238,188,186],
            [122,145,152,187,219,226,237,241,241,241,241,241,241,241,241,206,206,214,224,236,236,209,199,152],
            [69,86,95,101,114,131,157,158,158,159,159,159,159,159,159,154,131,137,148,154,156,108,92,115],
            [62,73,80,94,120,122,122,122,126,127,128,129,129,129,129,129,127,127,127,128,128,125,94,65],
            [53,80,91,101,111,118,117,117,119,119,119,119,119,119,119,119,109,109,109,112,116,95,83,68],
            [40,51,64,70,75,75,75,79,79,79,79,79,79,79,79,79,79,79,79,79,79,68,65,52],
            [13,26,30,44,51,52,52,54,55,55,55,55,55,55,55,42,52,53,54,54,54,52,43,34],
        ])

        self.pdm_hold = np.array(
            [1036, 1110, 1258, 1406, 1480, 1628, 1702, 1776, 1924, 2022, 2106, 2150, 2072, 1924, 1776, 1554, 1480, 1628, 1776, 1972, 1924, 1628, 1332, 1184])
        
        self.b = np.array([
            [0.000049, 0.000015, 0.000015, 0.000015, 0.000016, 0.000017, 0.000017, 0.000018, 0.000019, 0.000020],
            [0.000014, 0.000045, 0.000016, 0.000016, 0.000017, 0.000015, 0.000015, 0.000016, 0.000018, 0.000018],
            [0.000015, 0.000016, 0.000039, 0.000010, 0.000012, 0.000012, 0.000014, 0.000014, 0.000016, 0.000016],
            [0.000015, 0.000016, 0.000010, 0.000040, 0.000014, 0.000010, 0.000011, 0.000012, 0.000014, 0.000015],
            [0.000016, 0.000017, 0.000012, 0.000014, 0.000035, 0.000011, 0.000013, 0.000013, 0.000015, 0.000016],
            [0.000017, 0.000015, 0.000012, 0.000010, 0.000011, 0.000036, 0.000012, 0.000012, 0.000014, 0.000015],
            [0.000017, 0.000015, 0.000014, 0.000011, 0.000013, 0.000012, 0.000038, 0.000016, 0.000016, 0.000018],
            [0.000018, 0.000016, 0.000014, 0.000012, 0.000013, 0.000012, 0.000016, 0.000040, 0.000015, 0.000016],
            [0.000019, 0.000018, 0.000016, 0.000014, 0.000015, 0.000014, 0.000016, 0.000015, 0.000042, 0.000019],
            [0.000020, 0.000018, 0.000016, 0.000014, 0.000016, 0.000015, 0.000018, 0.000016, 0.000019, 0.000044],
        ])
        self.deltaDemands = []
        self.deltaDemands.append(0)
        for i in range(1, len(self.pdm_hold)):
            self.deltaDemands.append(self.pdm_hold[i] - self.pdm_hold[i - 1])
        #self.deltaDemands = self.pdm_hold[1:] - self.pdm_hold[:-1]
        self.demandRange = 1 + max(self.deltaDemands) + abs(min(self.deltaDemands))
        
        self.violation_multiplier = 1e6 # constant
        self.n_actions = 11
        self.previous_pnm = self.initial_pnm()
         # [Actions, Agents] transpose -> [Agents, Actions]
        self.action_powers = np.linspace(self.u_holder[:,0], self.u_holder[:,1], self.n_actions).T

        self.action_space = gym.spaces.MultiDiscrete(np.ones(len(self.u_holder)-1)*self.n_actions)
        self.observation_space = gym.spaces.MultiDiscrete(np.ones((len(self.u_holder)-1, 2))*20)
        self.reward_space = gym.spaces.Box(-np.ones(3)*np.inf, np.zeros(3))

    def initial_pnm(self):
        # start with average possible power
        return np.mean(self.u_holder[:,:2], axis=1)

    def reset(self):
        self.hour = 0
        self.previous_pnm = self.initial_pnm()
        #s = self.state(self.pdm_hold[0]-self.pdm_hold[-1], self.previous_pnm[1:])
        s = 0
        return s


    def get_bin(self, quantity, width, value):
        bins = np.arange(quantity+2)*width
        assert np.all(value >= bins[0]) and np.all(value <= bins[-1]), f'value {value} out of bounds for bins {bins}'
        # digitize gives the index of upper bound, provide lower bound
        bin_ = np.digitize(value, bins)-1
        return bin_

    """
    def discretize(self, pdm, power):
        # pdm
        quantity, width = 20, 25
        pdm_bin = self.get_bin(quantity, width, pdm)
        # power
        quantity, width = 20, 5
        print("Power", power)
        power_bin = self.get_bin(quantity, width, power)
        # make state for each agent, put the pdm with each power
        pdm_bin = np.array(pdm_bin).repeat(len(power_bin))
        # [Agent, State]
        return np.stack((pdm_bin, power_bin), axis=1)

    
    def state(self, pdm_delta, power):
        pdm_rescale = (pdm_delta - (-296)) * (492 - 0) / (196 - (-296)) + 0
        # state of all agents, not taking into account slack
        power_rescale = (power - self.u_holder[1:, 0])*(100-0)/(self.u_holder[1:, 1]-self.u_holder[1:, 0])+1
        current_state = self.discretize(pdm_rescale, power_rescale)
        return current_state

    """

    def discretize(self, state, bases):
        numState = bases[0] * bases[1]
        stateNo = 0
        #print(numState)
        #print(bases[0])
        #print(bases[1])
        #print(state)

        for i in range(len(state)):
            stateNo = stateNo * bases[i] + state[i]

        return stateNo

    def state(self, hour, _id_, _power_):
        state = 0

        if hour > 0 and hour <= 23:
            deltaDemand = self.deltaDemands[hour - 1] + abs(min(self.deltaDemands))
            genOffset = self.u_holder[_id_, 0]
            power = _power_ - genOffset * self.n_actions / (1 * (self.u_holder[_id_, 1] - self.u_holder[_id_, 0] - 1))
            #print("De;ta Demands", self.deltaDemands)
            #print(self.deltaDemands)
            #print(deltaDemand)
            #print(power)
            #print(12)
            state = self.discretize([deltaDemand, power], [self.demandRange, self.n_actions + 1])
        #print("State", state)

        return state

    def allowed_actions(self):
        max_power = np.clip(self.u_holder[:,12] + self.previous_pnm, self.u_holder[:,0], self.u_holder[:,1])
        min_power = np.clip(self.previous_pnm - self.u_holder[:,13], self.u_holder[:,0], self.u_holder[:,1])
        # for each agent, check actions whose power are below max, take their index, pick the last one
        max_action = [(self.action_powers[i] <= max_power[i]).nonzero()[0][-1] for i in range(len(max_power))]
        min_action = [(self.action_powers[i] >= min_power[i]).nonzero()[0][0] for i in range(len(max_power))]
        return np.stack((min_action, max_action), axis=1)

    def action_to_power(self, action):
        # [Agents, (min, max)]
        action_ranges = self.allowed_actions()
        # action_ranges without slack generator, clip between min and max
        corrected_actions = np.clip(action, action_ranges[1:,0], action_ranges[1:,1])
        # ignore slack generator, for each agent select the power corresponding to action index
        power = self.action_powers[np.arange(len(corrected_actions))+1, corrected_actions]
        return power

    def single_action_to_power(self, action, _id_):
        # [Agents, (min, max)]
        action_ranges = self.allowed_actions()
        # action_ranges without slack generator, clip between min and max
        corrected_actions = np.clip(action, action_ranges[_id_,0], action_ranges[_id_,1])
        # ignore slack generator, for each agent select the power corresponding to action index
        power = self.action_powers[_id_][corrected_actions]
        #print(power)
        return power


    def p1m(self, pnm, current_pdm):
        a = self.b[0, 0]
        # agents 1-N (ignore slack generator)
        n = np.arange(1,self.b.shape[1])
        # ! pnm does not contain slack power
        b = np.sum(self.b[0, n]*pnm)*2-1
        # repeat pnm to have same shape as b
        pnm_ = np.tile(pnm, (len(n), 1))
        sum_c_array = np.sum(pnm_*self.b[1:,1:]*pnm_)
        c = current_pdm + sum_c_array - np.sum(pnm)
        total_p1m = np.roots([a, b, c]).min()
        return total_p1m

    def fuel_cost(self, pnm):
        a, b, c = self.u_holder[:, 2], self.u_holder[:, 3], self.u_holder[:, 4]
        d, e, p_min = self.u_holder[:, 5], self.u_holder[:, 6], self.u_holder[:, 0]
        cost = a + b*pnm + c*pnm**2 + np.abs(d*np.sin(e*(p_min-pnm)))
        return cost

    def emission_cost(self, pnm, e=10):
        alpha, beta, gamma = self.u_holder[:, 7], self.u_holder[:, 8], self.u_holder[:, 9]
        eta, delta = self.u_holder[:, 10], self.u_holder[:, 11]
        cost = alpha + beta*pnm + gamma*pnm**2 + eta*np.exp(delta*pnm)
        return e*cost

    def ramp_violation(self, pnm, previous_pnm):
        pnm_diff = np.abs(pnm-previous_pnm)
        # assume dr==ur
        ur_dr_violation = pnm_diff > self.u_holder[:,12]
        violation = (pnm_diff-self.u_holder[:,12])*ur_dr_violation
        return violation

    def pnm_violation(self, pnm):
        max_violation = pnm > self.u_holder[:,1]
        min_violation = pnm < self.u_holder[:,0]
        violation = (pnm-self.u_holder[:,1])*max_violation + (self.u_holder[:,0]-pnm)*min_violation
        return violation

    def reward(self, previous_pnm, pnm, violation_multiplier=1e6):
        # for each agent, we'll get the different rewards
        fuel = self.fuel_cost(pnm)
        emission = self.emission_cost(pnm)
        # violations
        ramp = self.ramp_violation(pnm, previous_pnm)
        power = self.pnm_violation(pnm)
        violation = ramp + power
        violation[violation > 0] = (violation[violation > 0] + 1)*violation_multiplier
        # ! these are all costs, make them negative for rewards
        return -np.stack((fuel, emission, violation), axis=1)

    def step(self, state, _id_, previous_pnm, action, hour):
        # if next day, start over
        if hour == 0:
            previous_pnm = self.initial_pnm()

        _hour_ = (hour + 1) % 24
        pdm_delta = self.pdm_hold[hour] - self.pdm_hold[hour-1]
        
        #power = self.action_to_power(action)
        agent_power = self.single_action_to_power(action[_id_], _id_)
        action[_id_] = agent_power
        power = action
        # power for each agent, complete with slack generator
        p1m = self.p1m(power, self.pdm_hold[hour])
        pnm = np.concatenate((np.array([p1m]), power))

        #current_state = self.state(pdm_delta, power)
        current_state = self.state(hour, _id_, agent_power)
        # all the individual rewards are there, for each agent, *including* slack
        # [Agent, (fuel, emission, violation)]
        reward = self.reward(previous_pnm ,pnm, self.violation_multiplier)
        # print(f'{p1m} \t {reward[0]}')
        # TODO local, difference rewards
        reward = np.sum(reward, axis=0)

        previous_pnm = pnm
        done = _hour_ == 0
        return current_state, reward, pnm, done, {}


class FixedDEED(DEED):


    def __init__(self):
        super(FixedDEED, self).__init__()
        self.action_space = gym.spaces.Discrete(self.n_actions)


    def fixed_policies(self, hour):
        # always output the same power, half-capacity (50%)

        return np.ones(len(self.u_holder)-2, dtype=int)*50


    def step(self, _id_, action, hour):
        # first turbine is agent, other turbines use a fixed policy
        action = np.concatenate((np.array([action]), self.fixed_policies(hour)))
       
        return super(FixedDEED, self).step(_id_, action, hour)


class OptimalDEED(DEED):


    def __init__(self):
        super(OptimalDEED, self).__init__()
        self.action_space = gym.spaces.Discrete(self.n_actions)


    def optimal_policies(self, hour):
        # return the optimal power for each hour
        return self.generators[1:,hour]

    def step(self, state, _id_, pnm, action, hour):
        # _id_ turbine is agent, other turbines use a fixed optimal policy

        #action = np.concatenate((np.array([action]), self.optimal_policies(hour)))
        agents_power = self.optimal_policies(hour)
        agents_power[_id_] = action
        action = agents_power
        
        return super(OptimalDEED, self).step(state, _id_, pnm, action, hour)

# TODO : class DistributionDEED(DEED)
"""

if __name__ == '__main__':

    env = OptimalDEED()
    o = env.reset()
    d = False
    rew = 0
    hour = 0
    _id_ = 2
    for i in range(10):
        d = False
        hour = 0
        pnm = []
        #print("Timestep :", i)
        while not d:
            #action = env.action_space.sample()
            action = 0
            #print("Action", action)
            o, r, pnm, d, _ = env.step(0,_id_, pnm, action, hour)
            #print(o)
            rew += r
            
            #print(rew)
            if hour == 1:
                print("Hour", hour)
                print("Next State", o)
                print("Action", action)
                print("Reward", r)
            hour += 1
        #print("Overall Reward : ",rew)
"""
