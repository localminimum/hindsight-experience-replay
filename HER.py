#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf

from matplotlib import pyplot as plt
from tqdm import tqdm

# Bit flipping environment
class Env():
    def __init__(self, size = 8, shaped_reward = False):
        self.size = size
        self.shaped_reward = shaped_reward
        self.state = np.random.randint(2, size = size)
        self.target = np.random.randint(2, size = size)
        while np.sum(self.state == self.target) == size:
            self.target = np.random.randint(2, size = size)

    def step(self, action):
        self.state[action] = 1 - self.state[action]
        if self.shaped_reward:
            return self.state, -np.sum(np.square(self.state - self.target))
        else:
            if not np.sum(self.state == self.target) == self.size:
                return self.state, -1
            else:
                return self.state, 0

    def reset(self, size = None):
        if size is None:
            size = self.size
        self.state = np.random.randint(2, size = size)
        self.target = np.random.randint(2, size = size)

# Experience replay buffer
class Buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        # if len(self.buffer) + len(experience) >= self.buffer_size:
        #     overflow = len(self.buffer) + len(experience) - self.buffer_size
        #     self.buffer = self.buffer[-int(self.buffer_size - overflow):]
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[int(0.01 * self.buffer_size):]

    def sample(self,size):
        if len(self.buffer) >= size:
            experience_buffer = self.buffer
        else:
            experience_buffer = self.buffer * size
        return np.reshape(np.array(random.sample(experience_buffer,size)),[size,4])

# Simple 1 layer feed forward neural network
class Model():
    def __init__(self, size, name):
        with tf.variable_scope(name):
            self.size = size
            self.inputs = tf.placeholder(shape = [None, self.size * 2], dtype = tf.int32)
            self.unrolled = tf.reshape(tf.one_hot(self.inputs, depth = 2, axis = -1), [-1, 4 * self.size])
            self.fc = fully_connected_layer(self.unrolled, 256, activation = tf.nn.relu, scope = "fc1")
            self.Q_ = fully_connected_layer(self.fc, self.size, activation = None, scope = "readout")
            self.predict = tf.argmax(self.Q_, axis = -1)
            self.action = tf.placeholder(shape = None, dtype = tf.int32)
            self.action_onehot = tf.one_hot(self.action, self.size, dtype = tf.float32)
            self.Q = tf.reduce_sum(tf.multiply(self.Q_, self.action_onehot), axis = 1)
            self.Q_next = tf.placeholder(shape=None, dtype=tf.float32)
            self.loss = tf.reduce_sum(tf.square(self.Q_next - self.Q))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = self.optimizer.minimize(self.loss)
            self.init_op = tf.global_variables_initializer()

def fully_connected_layer(inputs, dim, activation = None, scope = "fc", reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        w_ = tf.get_variable("W_", [inputs.shape[-1], dim], initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b_", dim, initializer = tf.zeros_initializer())
        outputs = tf.matmul(inputs, w_) + b
        if activation is not None:
            outputs = activation(outputs)
        return outputs

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def main():
    HER = True
    shaped_reward = False
    size = 10
    num_epochs = 20
    num_cycles = 50
    num_episodes = 16
    optimisation_steps = 40
    K = 4
    buffer_size = 1e6
    tau = 1. - 0.95
    gamma = 0.98
    epsilon = 0.2
    batch_size = 128
    total_rewards = []
    succeed = 0

    modelNetwork = Model(size = size, name = "model")
    targetNetwork = Model(size = size, name = "target")
    trainables = tf.trainable_variables()
    updateOps = updateTargetGraph(trainables, tau)
    env = Env(size = size, shaped_reward = shaped_reward)
    buff = Buffer(buffer_size)
    with tf.Session() as sess:
        sess.run(modelNetwork.init_op)
        sess.run(targetNetwork.init_op)
        for i in tqdm(range(num_epochs), total = num_epochs):
            for j in range(num_cycles):
                total_reward = 0.0
                for n in range(num_episodes):
                    env.reset()
                    episode_experience = []
                    episode_succeeded = False
                    for t in range(size):
                        s = np.copy(env.state)
                        g = np.copy(env.target)
                        inputs = np.concatenate([s,g],axis = -1)
                        action = sess.run(modelNetwork.predict,feed_dict = {modelNetwork.inputs:[inputs]})
                        action = action[0]
                        if np.random.rand(1) < epsilon:
                            action = np.random.randint(size)
                        s_next, reward = env.step(action)
                        episode_experience.append((s,action,reward,s_next,g))
                        total_reward += reward
                        if reward == 0:
                            if episode_succeeded:
                                continue
                            else:
                                episode_succeeded = True
                                succeed += 1
                            # break
                        #     env.reset()
                        #     break
                    for t in range(len(episode_experience)):
                        s, a, r, s_n, g = episode_experience[t]
                        inputs = np.concatenate([s,g],axis = -1)
                        new_inputs = np.concatenate([s_next,g],axis = -1)
                        buff.add(np.reshape(np.array([inputs,a,r,new_inputs]),[1,4]))
                        if HER:
                            for k in range(K):
                                future = np.random.randint(t, size)
                                _, _, _, g_n, _ = episode_experience[future]
                                inputs = np.concatenate([s,g_n],axis = -1)
                                new_inputs = np.concatenate([s_n, g_n],axis = -1)
                                final = np.sum(np.array(s_n) == np.array(g_n)) == size
                                r_n = 0 if final else -1
                                buff.add(np.reshape(np.array([inputs,a,r_n,new_inputs]),[1,4]))

                for k in range(optimisation_steps):
                    experience = buff.sample(batch_size)
                    s, a, r, s_next = [np.squeeze(elem, axis = 1) for elem in np.split(experience, 4, 1)]
                    s = np.array([ss for ss in s])
                    s = np.reshape(s, (batch_size, size * 2))
                    s_next = np.array([ss for ss in s_next])
                    s_next = np.reshape(s_next, (batch_size, size * 2))
                    # Q1 = sess.run(modelNetwork.Q_, feed_dict = {modelNetwork.inputs: s})
                    Q1 = sess.run(modelNetwork.Q_, feed_dict = {modelNetwork.inputs: s_next})
                    Q2 = sess.run(targetNetwork.Q_, feed_dict = {targetNetwork.inputs: s_next})
                    doubleQ = Q2[:, np.argmax(Q1, axis = -1)]
                    # doubleQ = Q1[np.arange(batch_size),Q1]
                    Q_target = np.clip(r + gamma * doubleQ,  -1. / (1 - gamma), 0)
                    _ = sess.run(modelNetwork.train_op, feed_dict = {modelNetwork.inputs: s, modelNetwork.Q_next: Q_target, modelNetwork.action: a})
                updateTarget(updateOps,sess)
                total_rewards.append(total_reward)
    plt.plot(np.array(total_rewards))
    plt.show()
    print("Number of episodes succeeded: {}".format(succeed))

if __name__ == "__main__":
    main()
