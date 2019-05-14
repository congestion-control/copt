
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

        self.fc1 = tf.contrib.layers.fully_connected(self.state,64)
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
    uplink_trace = path.join(project_root.DIR, 'env', '114.68mbps.trace')
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


class Learner(object):
    def __init__(self, env):
        self.batch_size = 128
        self.y = 0.99
        self.tau = 0.001

        self.total_steps = 0
        self.num_episode = 1000
        self.max_epLength = 1000

        self.aug_state_dim = env.state_dim + env.action_cnt
        self.action_cnt = env.action_cnt
        self.prev_action = env.action_cnt - 1

        path = "./save_model"
        if not os.path.exists(path):
            os.makedirs(path)

        self.env = env

        self.state_dim = env.state_dim
        self.action_cnt = env.action_cnt

    def updateTargetGraph(self,tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars // 2]):
            op_holder.append(tfVars[idx + total_vars // 2].assign(
                (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
        return op_holder

    def updateTarget(self,op_holder, sess):
        for op in op_holder:
            sess.run(op)


    def cleanup(self):
        self.env.cleanup()



    def run(self):

        tf.reset_default_graph()
        self.mainQN = Q_network(state_dim=self.aug_state_dim, action_cnt=self.action_cnt)
        self.targetQN = Q_network(state_dim=self.aug_state_dim, action_cnt=self.action_cnt)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        trainables = tf.trainable_variables()
        targetOps = self.updateTargetGraph(trainables, self.tau)

        myBuffer = experience_buffer()
        self.rAll = 0
        self.jList = []
        self.rList = []
        F = open("r.txt", "w")

        with tf.Session() as sess:
            sess.run(init)

            def update_model():
                trainBatch = myBuffer.sample(self.batch_size)
                Q1 = sess.run(self.mainQN.predict, feed_dict={self.mainQN.state: np.vstack(trainBatch[:, 3])})
                Q2 = sess.run(self.targetQN.Qout, feed_dict={self.targetQN.state: np.vstack(trainBatch[:, 3])})

                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[xrange(self.batch_size), Q1]
                targetQ = trainBatch[:, 2] + (self.y * doubleQ * end_multiplier)

                _ = sess.run(self.mainQN.updateModel,
                                  feed_dict={self.mainQN.state: np.vstack(trainBatch[:, 0]),
                                             self.mainQN.targetQ: targetQ,
                                             self.mainQN.actions: trainBatch[:, 1]})

                self.updateTarget(targetOps, sess)

            def sample_action(state):
                if np.random.rand(1) < 0.05:
                    action = np.random.randint(0, self.env.action_cnt)
                else:

                    # Get probability of each action from the local network.
                    pi = self.mainQN
                    feed_dict = {
                        pi.state: [state],
                    }
                    ops_to_run = pi.predict
                    action = sess.run(ops_to_run, feed_dict)[0]

                    # Choose an action to take

                self.prev_action = action
                return action

            self.env.set_sample_action(sample_action)

            for episode_i in xrange(self.num_episode):
                sys.stderr.write('--- Episode %d\n' % episode_i)
                episode_buffer = experience_buffer()

                s = self.env.reset()

                # get an episode of experience
                buffer,rall = self.env.rollout()
                myBuffer.add(buffer.buffer)
                print(len(myBuffer.buffer))


                for i in xrange(2000):
                    #sys.stderr.write('update model %d\n' % i)
                    update_model()

                self.env.set_sample_action(sample_action)
                self.rList.append(rall)
                F.write(str(rall) + '\n')

                print('rall %f\n' % rall)
        F.close()
        return self.rList


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
    num_episode = 1000
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

    myBuffer = experience_buffer()
    rAll = 0
    jList = []
    rList = []
    F = open("r.txt", "w")
    with tf.Session() as sess:
        sess.run(init)

        def update_model():
            trainBatch = myBuffer.sample(self.batch_size)
            Q1 = sess.run(self.mainQN.predict, feed_dict={self.mainQN.state: np.vstack(trainBatch[:, 3])})
            Q2 = sess.run(self.targetQN.Qout, feed_dict={self.targetQN.state: np.vstack(trainBatch[:, 3])})

            end_multiplier = -(trainBatch[:, 4] - 1)
            doubleQ = Q2[xrange(self.batch_size), Q1]
            targetQ = trainBatch[:, 2] + (self.y * doubleQ * end_multiplier)

            _ = sess.run(self.mainQN.updateModel,
                         feed_dict={self.mainQN.state: np.vstack(trainBatch[:, 0]),
                                    self.mainQN.targetQ: targetQ,
                                    self.mainQN.actions: trainBatch[:, 1]})

            self.updateTarget(targetOps, sess)

        def sample_action(state):
            if np.random.rand(1) < 0.05:
                action = np.random.randint(0, self.env.action_cnt)
            else:

                # Get probability of each action from the local network.
                pi = self.mainQN
                feed_dict = {
                    pi.state: [state],
                }
                ops_to_run = pi.predict
                action = sess.run(ops_to_run, feed_dict)[0]

                # Choose an action to take

            self.prev_action = action
            return action

        self.env.set_sample_action(sample_action)

        for episode_i in xrange(self.num_episode):
            sys.stderr.write('--- Episode %d\n' % episode_i)
            episode_buffer = experience_buffer()

            s = self.env.reset()

            # get an episode of experience
            buffer, rall = self.env.rollout()
            myBuffer.add(buffer.buffer)
            print(len(myBuffer.buffer))

            for i in xrange(2000):
                # sys.stderr.write('update model %d\n' % i)
                update_model()

            self.env.set_sample_action(sample_action)
            self.rList.append(rall)
            F.write(str(rall) + '\n')

            print('rall %f\n' % rall)
    F.close()
    return self.rList




def main():


    env = create_env()
    learner = Learner(env)

    try:
        rlist = learner.run()


    except KeyboardInterrupt:
        pass
    finally:
        learner.cleanup()



if __name__ == '__main__':
    main()