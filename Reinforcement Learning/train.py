import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000    # how many iterations of the game to run
SHOW_EVERY = 10000   # show simulation every 1000 evolutions
STATS_EVERY = 100

# convert/bucket entire range into discrete values
# for example, 20 groups per range (can change this number)
DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
discreteOSWinSize = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
# discreteOSWinSize = how much to increment the range to get into the next bucket
#print(discreteOSWinSize)

# for displaying stats
epRewards = []
aggrEpRewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

# variable to encourage initial exploring
# decay epsilon every episode until done decaying it
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES/2
epsilonDecayValue = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# build the Q-table (with initialized random Q values):
# 20 x 20 x 3 --> 20 x 20 for every possible state, and x 3 for every possible action
# each step is a -1 reward, and the flag (win state) is a 0 reward, so make Q values all negative [-2, 0]
qTable = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

# cast state into one of the discrete buckets
def get_discrete_state(state):
    discreteState = (state - env.observation_space.low)/discreteOSWinSize
    return tuple(discreteState.astype(np.int))

for episode in range(EPISODES):
    render = True if episode % SHOW_EVERY == 0 else False
    episodeReward = 0
    
    discreteState = get_discrete_state(env.reset())
    done = False
    while not done:
        # encourages picking random value at start, then gradually use more argmax
        if np.random.random() > epsilon:
            action = np.argmax(qTable[discreteState])
        else:
            action = np.random.randint(0, env.action_space.n)
            
        # returns new state, the reward, whether the env is done (beat it or used 200 tries), and any extra info
        newState, reward, done, _ = env.step(action)
        newDiscreteState = get_discrete_state(newState)
        # accumulate reward for every action in the episode
        episodeReward += reward
        
        if render: env.render()
        
        # in Q-Learning, there's a table for each Q value per action per state
        # can query the environment to find possible state ranges:
        #print(env.observation_space.high)
        #print(env.observation_space.low)
        
        # to exploit environment, chooses action with highest Q value for the current state
        # if simulation didn't end, update the qTable:
        if not done:
            # best possible future action out of available choices
            qFutureMax = np.max(qTable[newDiscreteState])
            qCurrent = qTable[discreteState + (action,)]
            # calculate q value according to formula
            qNew = (1 - LEARNING_RATE) * qCurrent + LEARNING_RATE * (reward + DISCOUNT * qFutureMax)
            # update q value according to calculated value
            qTable[discreteState + (action,)] = qNew
        
        # if simulation ended and goal is achieved, update Q with reward
        # simulation can end after 200 rounds even if goal not achieved
        elif newState[0] >= env.goal_position:
            qTable[discreteState + (action,)] = 0   # where 0 is the reward
            
        # reset the discreteState variable for the next loop
        discreteState = newDiscreteState
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilonDecayValue
    
    # before the loop ends for that episode, store the reward
    epRewards.append(episodeReward)
    if not episode % STATS_EVERY:
        averageReward = sum(epRewards[-STATS_EVERY:])/STATS_EVERY
        aggrEpRewards['ep'].append(episode)
        aggrEpRewards['avg'].append(averageReward)
        aggrEpRewards['max'].append(max(epRewards[-STATS_EVERY:]))
        aggrEpRewards['min'].append(min(epRewards[-STATS_EVERY:]))
        
    # save every 100 qTable so can go back to earlier models
    # when opening new script, can open a table and continue to update those values or only lookup values
    if episode % 100 == 0:
        np.save(f"qtables\{episode}-qtable.npy", qTable)
    
env.close()
        
plt.plot(aggrEpRewards['ep'], aggrEpRewards['avg'], label = "average rewards")
plt.plot(aggrEpRewards['ep'], aggrEpRewards['max'], label = "max rewards")
plt.plot(aggrEpRewards['ep'], aggrEpRewards['min'], label = "min rewards")
plt.legend(loc = 4)
plt.grid(True)
plt.show()