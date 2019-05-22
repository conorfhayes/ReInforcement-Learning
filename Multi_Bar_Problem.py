
import numpy as np
import math
from random import randint
import time, os, fnmatch, shutil

class Agent:

    qTable = []
    actions = []

    def Agent(self, numActions, alpha, gamma, epsilon):
        self.numActions = numActions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        qTable, old_qTable = self.initialiseQvalue(numActions)
        self.qTable = qTable
        self.old_qTable = old_qTable
        self.selectedActions = []
        self.previousActions = []
        return self

    def decayEpsilon(self, epsilon):
        self.epsilon = epsilon * 0.95
        return self.epsilon

    def initialiseQvalue(self, numActions):
        qTable = np.zeros((numActions))
        old_qTable = qTable

        return qTable, old_qTable

    def updateQTable(self, previousState, selectedAction,reward):
        oldQ = self.old_qTable[previousState]
        maxQ = max(self.qTable)
        newQ = oldQ + self.alpha * (reward + self.gamma * maxQ - oldQ)
        self.old_qTable = self.qTable
        self.qTable[selectedAction] = newQ

        return self

    def selectAction(self):
        check = np.random.random()
        if check < self.epsilon:
            selectedAction = self.selectrandomAction()
            self.decayEpsilon(self.epsilon)
        else:
            selectedAction = self.getMaxValuedAction()

        return selectedAction

    def selectrandomAction(self):
        self.randomAction = round(randint(0,6))

        return self.randomAction

    def previousAction(self):
        previousAction = getMostRecentPreviousAction()
        self.previousActionValue = self.old_qTable[previousAction]

        return self.previousActionValue

    def previousActionTaken(self):
        self.previousAction = self.actions[len(self.actions)-2]

        return self.previousAction

    def getMaxValuedAction(self):
        self.maxIndex = np.argmax(self.qTable)

        return self.maxIndex

    def getMaxQValue(self):
        maxIndex = self.getMaxValuedAction()

        return self.qTable[maxIndex]

    def getQTable(self):

        return self.qTable

    def getOldQTable(self):

        return self.old_qTable

    def savePreviousActions(self, action):
        self.previousActions.append(action)
        return self.previousActions

    def saveSelectedActions(self, action):
        self.selectedActions.append(action)
        return self.selectedActions

    def getMostRecentAction(self):
        self.mostRecentAction = self.selectedActions[len(self.selectedActions)-1]
        return self.mostRecentAction

    def getMostRecentPreviousAction(self, k):
        if k == 0:
            return 0
        elif k == 1:
            return 0
        else:
            self.mostRecentPreviousAction = self.previousActions[len(self.previousActions)-2]
        return self.mostRecentPreviousAction


class Environment():

    global numActions
    global numAgents
    global _agents_
    global gamma
    global epsilon
    global alpha

    numActions = 7
    numEpisodes = 10000
    numAgents = 42
    epsilon = 0.1
    gamma = 1
    alpha = 0.1
    _agents_ = []
    a = Agent()

    def __init__(self):
        self.numActions = 7
        self.numEpisodes = 10000
        self.epsilon = 0.2
        self.gamma = 1
        self.alpha = 0.1
        self.k = 0
        self.t = time.localtime()
        self.timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', self.t)
        self.rewardArray = []

    def createAgents(self, numAgents):
        for i in range(numAgents):
            agent = Agent()
            agent_ = agent.Agent(numActions, alpha, gamma, epsilon)
            _agents_.append(agent_)

        return _agents_

    def getNextAction(self,action):
        if action == 0:
            nextAction == 0
        elif action == 1:
            nextAction = 1
        elif action == 2:
            nextAction == 2
        elif action == 3:
            nextAction = 3
        elif action == 4:
            nextAction = 4
        elif action == 5:
            nextAction = 5
        elif action == 6:
            nextAction = 6

        return nextAction

    def getBarPopulation(self,action, crowd):
        if action == 0:
            crowd[0] = crowd[0] + 1
        elif action == 1:
            crowd[1] = crowd[1] + 1
        elif action == 2:
            crowd[2] = crowd[2] + 1
        elif action == 3:
            crowd[3] = crowd[3] + 1
        elif action == 4:
            crowd[4] = crowd[4] + 1
        elif action == 5:
            crowd[5] = crowd[5] + 1
        elif action == 6:
            crowd[6] = crowd[6] + 1

        return crowd

    def calculateGlobalReward(self, pubCrowd, i):
        c = 6
        totalReward = []

        for people in pubCrowd:
            rewardPerDay = (math.exp(-(people/c))) * people
            totalReward.append(rewardPerDay)

        print("People Per Night", pubCrowd)
        print("Reward Per Night", totalReward)
        reward = sum(totalReward)
        print("Reward for Each Agent", reward)
        fileName = ("Multi_Bar_Problem_Global_Reward_" + self.timestamp + ".txt")
        line1 = '***************** Episode: ' + str(i) + ' ***********************'
        line2 = "People Per Night " + str(pubCrowd)
        line3 = "Reward Per Night " + str(totalReward)
        line4 = "Reward for Each Agent " + str(reward)

        with open(fileName, 'a') as out:
            out.write('{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4))
        out.close()

        return reward

    def calculateLocalReward(self,action_, pubCrowd, i):
        c = 6
        totalReward = []

        for people in pubCrowd:
            rewardPerDay = (math.exp(-(people/c))) * people
            totalReward.append(rewardPerDay)

        #print("People Per Night", pubCrowd)
        #print("Reward Per Night", totalReward)

        if action_ == 0:
            reward = totalReward[0]
        elif action_ == 1:
            reward = totalReward[1]
        elif action_ == 2:
            reward = totalReward[2]
        elif action_ == 3:
            reward = totalReward[3]
        elif action_ == 4:
            reward = totalReward[4]
        elif action_ == 5:
            reward = totalReward[5]
        elif action_ == 6:
            reward = totalReward[6]

        #print("Reward for Each Agent", reward)
        fileName = ("Multi_Bar_Problem_Local_Reward_" + self.timestamp + ".txt")
        line1 = '***************** Episode: ' + str(self.k) + ' ***********************'
        line2 = 'Agent: ' + str(i)
        line3 = 'Agent ' + str(i) + " Action: " + str(action_)
        line4 = "People Per Night " + str(pubCrowd)
        line5 = "Reward Per Night " + str(totalReward)
        line6 = "Reward for Each Agent " + str(reward)

        with open(fileName, 'a') as out:
            out.write('{}\n{}\n{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4, line5, line6))
        out.close()

        return reward

    def OverallReward(self, reward):
        self.rewardArray.append(reward)

    def getOverallReward(self):
        return self.rewardArray


    def calculateDifferenceReward(self,action_, pubCrowd, i, agent):
        c = 6
        totalReward = []

        for people in pubCrowd:
            rewardPerDay1 = (math.exp(-(people/c))) * people
            rewardPerDay2 = (math.exp(-((people - 1)/c))) * (people - 1)
            rewardPerDay = rewardPerDay1 - rewardPerDay2
            totalReward.append(rewardPerDay)

        #print("People Per Night", pubCrowd)
        #print("Reward Per Night", totalReward)

        if action_ == 0:
            reward = totalReward[0]
        elif action_ == 1:
            reward = totalReward[1]
        elif action_ == 2:
            reward = totalReward[2]
        elif action_ == 3:
            reward = totalReward[3]
        elif action_ == 4:
            reward = totalReward[4]
        elif action_ == 5:
            reward = totalReward[5]
        elif action_ == 6:
            reward = totalReward[6]

        #print("Reward for Each Agent", reward)
        fileName = ("Multi_Bar_Problem_Difference_Reward_" + self.timestamp + ".txt")
        line1 = '***************** Episode: ' + str(self.k) + ' ***********************'
        line2 = 'Agent: ' + str(i)
        line3 = 'Agent ' + str(i) + " Action: " + str(action_)
        line4 = "People Per Night " + str(pubCrowd)
        line5 = "Reward Per Night " + str(totalReward)
        line6 = "Reward for Each Agent " + str(reward)
        line7 = "Q Table " + str(agent.getQTable())

        with open(fileName, 'a') as out:
            out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4, line5, line6, line7))
        out.close()
        self.OverallReward(reward)
        return reward

    def step(self, _agents_, i):
        crowd = np.zeros(numActions)
        selectedActions = []
        self.k = self.k + 1
        for agent in _agents_:
            action_ = agent.selectAction()
            #("Action: ", action_)
            agent.saveSelectedActions(action_)
            selectedActions.append(action_)
            previousAction = agent.getMostRecentPreviousAction(self.k)
            agent.savePreviousActions(previousAction)

        for action in selectedActions:
            population = self.getBarPopulation(action, crowd)

        # Uncomment Line Below to Use Global Reward
        reward = self.calculateGlobalReward(population, i)

        i = 0
        for agent in _agents_:
            action_ = agent.getMostRecentAction()

            # Uncomment Line Below to Use Local Reward
            #reward = self.calculateLocalReward(action_,population, i)

            # Uncomment Line Below to Use Difference Reward
            #reward = self.calculateDifferenceReward(action_, population, i, agent)

            agent.getQTable = agent.getOldQTable
            agent.updateQTable(int(agent.getMostRecentPreviousAction(self.k)), int(agent.getMostRecentAction()), reward)
            i = i + 1

def main():
    numEpisodes = 4000
    numAgents = 42

    env = Environment()

    _agents_ = env.createAgents(numAgents)

    global fileName

    i = 1
    while i < numEpisodes:
        print("Episode:", i)
        env.step(_agents_, i)

        i = i + 1
    print(env.getOverallReward())

if __name__ == "__main__":
    main()
