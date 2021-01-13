from keras.callbacks import TensorBoard

class ModifiedTensorBoard(TensorBoard):
    # overriding __init__ to only write in one log file for all .fit()'s
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # stop creating default log writer
    def set_model(self, model):
        pass

    # only saves logs every n steps instead of every step from start
    def on_epoch_end(self, epoch, logs = None):
        self.update_state(**logs)

    # only plan to train for one batch, so don't save at epoch end
    def on_batch_end(self, batch, logs = None):
        pass

    # don't close writer
    def on_train_end(self, _):
        pass

    # creates writer, writes custom metrics, then closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:
    def create_model(self):
        model = Sequential()

        # Note: OBSERVATION_SPACE_VALUES = (10, 10, 3); RGB image
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
        model.compilt(loss = "mse", optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
        return model

    def __init_(self):
        ''' 
        initially, if model starts off as random and gets updated every step, 
        every episode, it'll have big & confusing fluctuations
        Instead, have two models with "memory replay"
        Eventually, converge the two models. However, want the model that makes
        predictions to be more stable than the one that gets fitted/updated
        '''

        # main model, used to make predictions
        self.model = self.create_model()

        # target network: gets updated and decides on future Q values
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # array with the last n steps for training
        # remember the previous 1000 actions, then fit model on a random selection from these actions
        # helps smooth out fluctuations
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

        # in unmodified TensorBoard, keras writes a log file per .fit() which generates a lot of data
        self.tensorboard = ModifiedTensorBoard(log_dir = "logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # count when to update target network with main network's weights
        self.target_update_counter = 0

    # add data from the current step to the end of memory replay
    # values: (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # given the current env state (obs space), predict a Q value
    # Note: reshape array because TensorFlow wants that exact shape
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]