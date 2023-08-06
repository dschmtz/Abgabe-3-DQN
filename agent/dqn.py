import random
import numpy as np
import tensorflow as tf
from collections import deque


class DQN:

    def __init__(self, action_space, state_shape, learning_rate=0.001, 
                 gamma=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, batch_size=16, memory_size=2000):
        self.action_space = action_space
        self.state_shape = state_shape

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)
        self.model = self.build_model()

    def build_model(self):
        input = tf.keras.Input(shape=(self.state_shape))

        # Feature learning of graphical properties of the environment
        backbone = tf.keras.layers.Conv2D(8, kernel_size=(7, 7), strides=3)(input)
        backbone = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(backbone)
        backbone = tf.keras.layers.Conv2D(16, kernel_size=(3, 3))(backbone)
        backbone = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(backbone)

        # Head for classifying the action
        head = tf.keras.layers.Flatten()(backbone)
        head = tf.keras.layers.Dense(128, activation="relu")(head)

        output = tf.keras.layers.Dense(len(self.action_space), activation="linear", name="actions")(head)

        model = tf.keras.Model(inputs=input, outputs=output)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        action_ind = self.action_space.index(action)
        self.memory.append((state, action_ind, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Exploration: Chosing random actions
            action_ind = random.randrange(len(self.action_space))
        else:
            # Exploitation: Use modell prediction
            pred = self.model.predict(np.expand_dims(state, axis=0), verbose=0) 
            action_ind = np.argmax(pred[0])

        return self.action_space[action_ind]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a random mini-batch from experience replay memory
        mini_batch = random.sample(self.memory, self.batch_size)

        states = []
        targets = []
        for state, action_ind, reward, next_state, done in mini_batch:
            q_target = reward
            
            if not done:
                # Use the model to estimate the next Q-value
                next_state = np.expand_dims(state, axis=0)
                next_actions = self.model.predict(next_state, verbose=0)[0]
                q_target += self.gamma * np.amax(next_actions)

            # Use the model to get the current Q-values and update
            expanded_state = np.expand_dims(state, axis=0)
            cur_actions = self.model.predict(expanded_state, verbose=0)[0]
            cur_actions[action_ind] = q_target

            states.append(state)
            targets.append(cur_actions)

        states = np.array(states)
        targets = np.array(targets)

        # Train the model on the mini-batch   
        result = self.model.train_on_batch(states, targets)

        # Decay exploration rate epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return result
    