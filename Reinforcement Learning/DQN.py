# instead of a Q table to look up values, use NN to make predictions
# input states, output Q values for each possible action
# use .predict() to find next move based on Q values
# then update network by .fit() based on updated Q values
    # fit all three, even if only intend to update one

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
from blob_env import *

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000      # how big of memory to keep for training
MIN_REPLAY_MEMORY_SIZE = 1000   # minimum number of steps stored in memory to start training
MINIBATCH_SIZE = 64             # how many steps/samples to use for training (1 fit)
UPDATE_TARGET_EVERY = 5         # update target model every 5 terminal states (end of episodes)
MODEL_NAME = "2x256"
MIN_REWARD = -200               # only save model with at least reward = -200
MEMORY_FRACTION = 0.20

# environment settings
EPISODES = 20000

# exploration settings
epsilon = 1                     # how much to explore instead of going greedy
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# stats settings
AGGREGATE_STATS_EVERY = 50      # aggregate every 50 episodes
SHOW_PREVIEW = False

# own TensorBoard class which saves logs less often
class ModifiedTensorBoard(TensorBoard):
    # overriding __init__ to only write in one log file for all .fit()'s
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.model = None
        self.TB_graph = tf.compat.v1.Graph()
        with self.TB_graph.as_default():
            self.writer = tf.summary.create_file_writer(self.log_dir, flush_millis=5000)
            self.writer.set_as_default()
            self.all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
        self.TB_sess = tf.compat.v1.InteractiveSession(graph=self.TB_graph)
        self.TB_sess.run(self.writer.init())

    # stop creating default log writer
    def set_model(self, model):
        self.model = model
        self._train_dir = self.log_dir + '\\train'

    # only saves logs every n steps instead of every single step since start
    def on_epoch_end(self, epoch, logs = None):
        self.update_state(**logs)

    # only plan to train for one batch, so don't save at epoch end
    def on_batch_end(self, batch, logs = None):
        pass

    def on_train_begin(self, logs=None):
        pass

    # don't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, _, __):
        pass

    # creates writer, writes custom metrics, then closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            self.TB_sess.run(self.all_summary_ops)
            if self.model is not None:
                name = f'{name}_{self.model.name}'
            self.TB_sess.run(tf.summary.scalar(name, value, step=index))
        self.model = None

class DQNAgent:
    def __init__(self):
        ''' 
        initially, if model starts off as random and gets updated every step, 
        every episode, it'll have big & confusing fluctuations
        Instead, have two models with "memory replay"
        Eventually, converge the two models. However, want the model that makes
        predictions to be more stable than the one that gets fitted/updated
        '''

        # main model --> gets trained on minibatches
        self.model = self.create_model()

        # target network --> used to make predictions
            # gets updated by weights from main model every UPDATE_TARGET_EVERY
        # by not getting trained at the same time of making predictions,
            # target model is more stable than main, especially at the beginning
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # array with the last n steps for training
        # remember the previous 1000 actions, then fit model on a random selection from these actions
        # helps smooth out fluctuations
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

        # in unmodified TensorBoard, keras writes a log file per .fit() 
        # which generates a lot of data
        self.tensorboard = ModifiedTensorBoard(log_dir = "logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        # Note: OBSERVATION_SPACE_VALUES = (10, 10, 3); RGB image
        # consists of the board and the 3 items: player, food, and enemy
        model.add(Conv2D(256, (3, 3), input_shape = env.OBSERVATION_SPACE_VALUES))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        # Note: ACTION_SPACE_SIZE = number of choices = 9
        model.add(Dense(env.ACTION_SPACE_SIZE, activation = 'linear'))
        model.compile(loss = "mse", optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
        return model

    # add data from the current step to the end of memory replay
    # values: (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # given the current env state (obs space), predict a Q value
    # note: reshape array because TensorFlow wants that exact shape
    # note: transition[0] contains the observation space
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    # trains main network every step during episode
    def train(self, terminal_state, step):
        # start training only if a certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # if do have enough memory, randomly select training set from memory
        # get minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # get current states from minibatch, then query model for Q values
        # recall: transition[0] contains the observation space
            # so current_states is an array of observation spaces from the minibatch
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # get future states from minibatch, then query model for Q values
        # query target network if using it, else query main network
            # recall: separating prediction network and training network makes predictions more stable
        # recall: transition[3] contains the new observation space for that transition memory
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        # get ready to upate the model
        X = []
        y = []
        
        # enumerate through each example in the minibatch
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # if not a terminal state, get new Q from future states
            # else set it to 0
            # similar to equation for Q-learning
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # update Q values for the given state
            # recall: current_qs_list comes from model.predict(current_states)
            current_qs = current_qs_list[index] # for this example in the minibatch, get q's for each action
            current_qs[action] = new_q          # update the action you took with the new calculated q

            # append to training data
            X.append(current_state) # input: observation space
            y.append(current_qs)    # labels: updated q values given the observation space

        # after generating inputs (observation states) and labels (q values) for every example in the minibatch,
        # fit the model on all samples as one batch
        # only log the terminal state
        self.model.fit(np.array(X)/255, np.array(y), 
            batch_size = MINIBATCH_SIZE, verbose = 1, shuffle = False, 
            callbacks = [self.tensorboard] if terminal_state else None)

        # update target network update counter only if done == True
        if terminal_state:
            self.target_update_counter += 1

        # if counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

# initialize environment
env = BlobEnv()

# for stats
ep_rewards = [-200]

# for repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# memory fraction used when training multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = MEMORY_FRACTION)
#backend.set_session(tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)))

# create models folder
if not os.path.isdir('blob_models'):
    os.makedirs('blob_models')

# use agent to engage with the environment
agent = DQNAgent()

# iterate over episodes
# wrap iterable with tqdm to show progress bar
for episode in tqdm(range(1, EPISODES + 1), ascii = True, unit = 'episodes'):
    # update tensorboard step every episode
    agent.tensorboard.step = episode

    # restart episode by resetting episode reward and step number
    episode_reward = 0
    step = 1

    # reset environment and get initial state
    current_state = env.reset()

    # after restarting episode and state, iterate over steps in the episode:
    done = False
    while not done:
        # if greedy, choose action based on max Q value given current state
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        # else choose a random action (out of the 9 available)
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        # update information after the action was applied
        # if done becomes true, the loop exits next round
        new_state, reward, done = env.step(action)

        episode_reward += reward

        # if preview is turned on and it's the n'th episode, render the scene
        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # update replay memory and train the main network
            # agent.train() also trains target network if step size is right
        # change current_state from pre-action to post-action
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        current_state = new_state
        step += 1
    
    # log stats every so often and save model if it's performing well enough
    # check that average reward is above some number
    ep_rewards.append(episode_reward)   # save final reward of episode to array
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        # average over the last AGGREGATE_STATS_EVERY episodes
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg = average_reward, 
            reward_min = min_reward, reward_max = max_reward, epsilon = epsilon)

        # save model, but only when min reward is above some threshold
        if min_reward >= MIN_REWARD:
            agent.model.save(f'blob_models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # decay epsilon after end of every episode
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon) # make sure not lower than min        