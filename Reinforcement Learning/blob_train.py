import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from blob_object import *

style.use("ggplot")

SIZE = 10               # number of bucket divisions
HM_EPISODES = 50000     # number of episodes in total
MOVE_PENALTY = 1        # tune - costs energy to move
ENEMY_PENALTY = 300     # tune - avoid touching the enemy
FOOD_REWARD = 25        # tune - food is good
epsilon = 1.0           # how often to randomize decisiosn
EPS_DECAY = 0.9998      # epsilon multiplies to decay each episode
SHOW_EVERY = 5000       # how often to display the game

#startQTable = "qtable-1610343507.pickle"      # make a new table instead of using a saved one
startQTable = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_C = 1            # player key in dict
FOOD_C = 2              # food key in dict
ENEMY_C = 3             # enemy key in dict

colourDict = {1: (255, 175, 0),  # player = blue
              2: (0, 255, 0),    # food = green
              3: (0, 0, 255)}    # enemy = red

# if generating a new Q table
if startQTable is None:
    qTable = {}         # initialize the qTable; x and y to both food and enemy
    for i in range(-SIZE + 1, SIZE):
        for ii in range(-SIZE + 1, SIZE):
            for iii in range(-SIZE + 1, SIZE):
                for iiii in range(-SIZE + 1, SIZE):
                    qTable[((i, ii), (iii, iiii))] = [
                        np.random.uniform(-5, 0) for i in range(4)]

# if reading a Q table from file
else:
    with open(startQTable, "rb") as f:
        qTable = pickle.load(f)

episodeRewards = []

for episode in range(HM_EPISODES):
    # initialize new player, food, and enemy for each episode
    player = Blob(SIZE)
    food = Blob(SIZE)
    enemy = Blob(SIZE)

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        show = True
    else:
        show = False

    episodeReward = 0
    for i in range(200):    # 200 action items per episode

        # choosing an action and moving the pieces
        obs = (player - food, player - enemy)   # current observation

        if np.random.random() > epsilon:        # if don't randomize
            action = np.argmax(qTable[obs])     # then pick the best action
        else:                                   # if do randomize
            action = np.random.randint(0, 4)    # then choose a random action

        player.action(action)   # player takes the chosen action
        enemy.move()            # no direction given, so moves randomly
        food.move()
        # note: a trained stationary model can quickly adapt to a moving model
        # even if moved from start, still trains well

        # assigning rewards
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        # assess outcome of the chosen action
        newObs = (player - food, player - enemy)    # new observation
        # max Q possible for the next decision
        maxFutureQ = np.max(qTable[newObs])
        # current Q for chosen action
        currentQ = qTable[obs][action]

        # calculate and apply rewards
        if reward == FOOD_REWARD:
            newQ = FOOD_REWARD
        else:
            newQ = (1 - LEARNING_RATE) * currentQ + \
                LEARNING_RATE * (reward + DISCOUNT * maxFutureQ)
        qTable[obs][action] = newQ

        # visualize what the model is doing
        if show:
            # starts rbg of set size
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.x][food.y] = colourDict[FOOD_C]           # sets colours
            env[player.x][player.y] = colourDict[PLAYER_C]
            env[enemy.x][enemy.y] = colourDict[ENEMY_C]
            img = Image.fromarray(env, 'RGB')      # create image from array
            img = img.resize((300, 300))           # resize image
            cv2.imshow("image", np.array(img))     # show image

            # if code finishes before the episode finishes
            # note: & 0xFF is a bit mask which gets the last 8 bits of cv2.waitKey
            # cv2.waitKey waits __ ms for a keyboard response
            # if the letter 'q' is pressed while waiting for a key, it breaks the loop and goes on to the next episode
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episodeReward += reward

        # break episode if reached objective or hit enemy
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    # after accumulating all rewards from this episode
    episodeRewards.append(episodeReward)
    epsilon *= EPS_DECAY

# in this convolution, a mask of size SHOW_EVERY with each value = 1/SHOW_EVERY is applied on episodeRewards
# essentially, movingAvg = (1/SHOW_EVERY) * SUM(episodeRewards[start:start + SHOW_EVERY])
# where episodeRewards range sweeps over the whole array
# resulting in an array of moving averages across the episodeRewards array
movingAvg = np.convolve(episodeRewards, np.ones(
    (SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(movingAvg))], movingAvg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

# write the final qTable into a pickle file
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(qTable, f)
