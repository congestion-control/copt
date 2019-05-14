import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
import os
import sys
from env.sender import Sender
from helpers.helpers import normalize, one_hot, softmax
from environment import Environment
import random
import tensorflow.contrib.slim as slim


class Q_network(object):
    def __init__(self, state_dim, action_cnt):
        self.state = tf.placeholder(shape=[None, state_dim], dtype=tf.float32)
        self.state = tf.reshape(self.state, shape=[-1, state_dim])

        self.fc1 = tf.contrib.layers.fully_connected(self.state, 64)
        self.fc1 = tf.nn.dropout(self.fc1, 0.8)

        self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 64)
        self.fc2 = tf.nn.dropout(self.fc2, 0.5)

        self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 64)
        self.fc3 = tf.nn.dropout(self.fc3, 0.5)

        self.streamAC, self.streamVC = tf.split(self.fc3, 2, 1)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)

        self.Advantage = tf.contrib.layers.fully_connected(self.streamA, action_cnt)
        self.Value = tf.contrib.layers.fully_connected(self.streamV, 1)

        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True))
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_cnt, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.updateModel = self.trainer.minimize(self.loss)


def create_env():
    uplink_trace = path.join(project_root.DIR, 'env', '77.72mbps.trace')
    downlink_trace = uplink_trace
    mahimahi_cmd = (
        'mm-delay 20 mm-link %s %s '
        '--downlink-queue=droptail --downlink-queue-args=packets=200' %
        (uplink_trace, downlink_trace))

    env = Environment(mahimahi_cmd)
    return env


class experience_buffer():
    def __init__(self, buffer_size=100000):
        self.buffer = []

        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])





def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def cleanup(env):
    env.cleanup()


def run_learner(env):
    batch_size = 128
    y = 0.99
    tau = 0.001

    total_steps = 0
    num_episode = 1
    max_epLength = 1000

    aug_state_dim = env.state_dim + env.action_cnt
    action_cnt = env.action_cnt
    prev_action = env.action_cnt - 1

    path = "./save_model"
    if not os.path.exists(path):
        os.makedirs(path)

    state_dim = env.state_dim
    action_cnt = env.action_cnt

    tf.reset_default_graph()
    mainQN = Q_network(state_dim=aug_state_dim, action_cnt=action_cnt)
    targetQN = Q_network(state_dim=aug_state_dim, action_cnt=action_cnt)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables, tau)

    #myBuffer = experience_buffer()
    rAll = 0
    jList = []
    rList = []

    with tf.Session() as sess:
        sess.run(init)

        def update_model(myBuffer):
            trainBatch = myBuffer.sample(batch_size)
            Q1 = sess.run(mainQN.predict, feed_dict={mainQN.state: np.vstack(trainBatch[:, 3])})
            Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.state: np.vstack(trainBatch[:, 3])})

            end_multiplier = -(trainBatch[:, 4] - 1)
            doubleQ = Q2[xrange(batch_size), Q1]
            targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)

            _ = sess.run(mainQN.updateModel,
                         feed_dict={mainQN.state: np.vstack(trainBatch[:, 0]),
                                    mainQN.targetQ: targetQ,
                                    mainQN.actions: trainBatch[:, 1]})

            updateTarget(targetOps, sess)

        def sample_action(state):
            if np.random.rand(1) < 0.05:
                action = np.random.randint(0, env.action_cnt)
            else:

                # Get probability of each action from the local network.
                pi = mainQN
                feed_dict = {
                    pi.state: [state],
                }
                ops_to_run = pi.predict
                action = sess.run(ops_to_run, feed_dict)[0]

                # Choose an action to take

            prev_action = action
            return action

        env.set_sample_action(sample_action)
        env.set_update_Qnet(update_model)

        for episode_i in xrange(num_episode):
            sys.stderr.write('--- Episode %d\n' % episode_i)

            s = env.reset()

            # get an episode of experience
            env.rollout()


    F.close()
    return rList


def main():
    env = create_env()


    try:
        run_learner(env)


    except KeyboardInterrupt:
        pass
    finally:
        cleanup(env)


if __name__ == '__main__':
    main()