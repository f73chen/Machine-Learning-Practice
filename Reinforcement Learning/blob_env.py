from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
#import keras.backend.tensorflow_backend as backend
import numpy as np
import time
import random
import os
import cv2
from blob_object import *

class BlobEnv:
    SIZE = 10               # number of bucket divisions
    RETURN_IMAGES = True
    MOVE_PENALTY = 1        # tune - costs energy to move
    ENEMY_PENALTY = 300     # tune - avoid touching the enemy
    FOOD_REWARD = 25        # tune - food is good
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    ACTION_SPACE_SIZE = 9

    PLAYER_C = 1            # player key in dict
    FOOD_C = 2              # food key in dict
    ENEMY_C = 3             # enemy key in dict

    colourDict = {1: (255, 175, 0), # player = blue
                2: (0, 255, 0),     # food = green
                3: (0, 0, 255)}     # enemy = red

    def reset(self):
        self.player = Blob(self.SIZE)   # initialize random position of player
        self.food = Blob(self.SIZE)     # initialize random position of food
        self.enemy = Blob(self.SIZE)    # initialize random position of enemy
        
        # re-initialize food and enemy locations if overlapping player
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        # reset number of steps taken this episode
        self.episode_step = 0

        # whether to use image-based (CNN) or distance-based observations
        if self.RETURN_IMAGES:
            # returns a SIZE x SIZE cv2 image for CNN to pass over
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food) + (self.player - self.enemy)
        return observation

    # during each step, execute the action, make new observation, apply rewards, and assess if the episode is done
    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        ''' MAYBE
        enemy.move()
        food.move()
        '''
        
        # get new observation space after making the move
        if self.RETURN_IMAGES:
            # returns a SIZE x SIZE cv2 image for CNN to pass over
            new_observation = np.array(self.get_image())
        else:
            # else use distance-based observation
            new_observation = (self.player - self.food) + (self.player - self.enemy)

        # apply rewards based on state
        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        # episode not done until player reached goal, died, or reached step limit for that episode
        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        # after performing the action, return new observation space, cumulative reward score, and whether the episode is over
        return new_observation, reward, done

    # draws the episode with OpenCV for the user to see
    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))
        cv2.imshow("image", np.array(img))
        cv2.waitKey(1)  # refresh render every 1 millisecond

    # returns image object for CNN to pass over
    def get_image(self):
        # starts RGB window of set size
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype = np.uint8)

        # sets food, enemy, and player locations to corresponding colours
        env[self.food.x][self.food.y]     = self.colourDict[self.FOOD_C]
        env[self.enemy.x][self.enemy.y]   = self.colourDict[self.ENEMY_C]
        env[self.player.x][self.player.y] = self.colourDict[self.PLAYER_C]

        # reading to RGB even though colour definitions are BGR??
        img = Image.fromarray(env, 'RGB')
        return img