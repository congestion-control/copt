#adjusted from https://github.com/StanfordSNR/indigo/blob/master/env/sender.py
#Minhui Huang

import time
import sys
import json
import socket
import select
from os import path
import numpy as np
import datagram_pb2
import project_root
import random
from helpers.helpers import (
    curr_ts_ms, normalize,one_hot,
    READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, WRITE_FLAGS, ALL_FLAGS)



class copt_Sender(object):
    #initialization

    def __init__(self, port=0, debug=False):
        self.debug = debug

        self.interval_rate = 0

        self.segs_sent_start = 0
        self.segs_sent_end = 0

        self.utility = 0
        self.action = 0
        self.lost = 0
        self.t = 0.9

        #UDP socket and poller
        self.peer_addr = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # The setsockopt() function shall set the option specified by the option_name argument,
        # SO_REUSEADDR: Specifies that the rules used in validating addresses supplied to bind() should allow reuse of local addresses,
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #Bind the socket to address.
        self.sock.bind(('0.0.0.0', port))

        #getsockname : Return the socket's own address
        sys.stderr.write('[sender] Listening on port %s\n' %self.sock.getsockname()[1])

        # Returns a polling object, which supports registering and unregistering file descriptors, and then polling them for I/O events
        self.poller = select.poll()
        #Register a file descriptor with the polling object.
        self.poller.register(self.sock, ALL_FLAGS)

        self.MSS = 1400
        self.dummy_payload = '*' * self.MSS

        #congestion control related
        self.seq_num = 0
        self.next_ack = 0
        self.cwnd = 10.0
        self.step_len_ms = 10

        #state variable for RLCC
        self.delivered_time = 0
        self.delivered = 0
        self.sent_bytes = 0

        self.min_rtt = float('inf')

        self.delay_ewma = None
        self.send_rate_ewma = None
        self.delivery_rate_ewma = None

        self.step_start_ms = None
        self.running = True

    def cleanup(self):
        self.sock.close()

    def handshake(self):
        #handshake with peer receiver. Must be called before run()

        while True:

            #Receive data from the socket. The return value is a pair (string, address)
            # where string is a string representing the data received and address is the address of the socket sending the data.
            msg, addr = self.sock.recvfrom(1600)
            #print(msg)

            if msg == 'Hello from receiver' and self.peer_addr == None:
                self.peer_addr = addr
                self.sock.sendto('Hello from sender', self.peer_addr)
                sys.stderr.write('[sender] Handshake success! '
                                 'Receiver\'s address is %s:%s\n' % addr)
                break
        #setblocking(0):Set blocking or non-blocking mode of the socket: if flag is 0, the socket is set to non-blocking, else to blocking mode
        self.sock.setblocking(0) # non-blocking UDP socket



    def update_state(self, ack):

        self.next_ack = max(self.next_ack, ack.seq_num + 1)
        curr_time_ms = curr_ts_ms()

        rtt = float(curr_time_ms - ack.send_ts)
        self.min_rtt = min(self.min_rtt, rtt)

        queue_delay = rtt - self.min_rtt

        if self.delay_ewma is None:
            self.delay_ewma = queue_delay
        else:
            self.delay_ewma = 0.875 * self.delay_ewma + 0.125 * queue_delay


        # Update BBR's delivery rate
        self.delivered += ack.ack_bytes
        self.delivered_time = curr_time_ms
        delivery_rate = (0.008 * (self.delivered - ack.delivered) /
                         max(1, self.delivered_time - ack.delivered_time))

        if self.delivery_rate_ewma is None:
            self.delivery_rate_ewma = delivery_rate
        else:
            self.delivery_rate_ewma = (
                0.875 * self.delivery_rate_ewma + 0.125 * delivery_rate)


        # Update Vegas sending rate
        send_rate = 0.008 * (self.sent_bytes - ack.sent_bytes) / max(1, rtt)

        if self.send_rate_ewma is None:
            self.send_rate_ewma = send_rate
        else:
            self.send_rate_ewma = (
                0.875 * self.send_rate_ewma + 0.125 * send_rate)

    def adjust_sending_rate(self):
        old_rate = self.interval_rate
        new_rate = np.power(self.t/(1000 * self.delay_ewma + 900 * self.lost), 1/(1-self.t))
        self.interval_rate = new_rate


    def window_is_open(self):
        return (self.segs_sent_end - self.segs_sent_start)*self.MSS < self.interval_rate

    def start_interval(self):


    def send(self):
        data = datagram_pb2.Data()
        data.seq_num = self.seq_num
        data.send_ts = curr_ts_ms()
        data.sent_bytes = self.sent_bytes
        data.delivered_time = self.delivered_time
        data.delivered = self.delivered
        data.payload = self.dummy_payload

        serialized_data = data.SerializeToString()
        self.sock.sendto(serialized_data, self.peer_addr)

        self.seq_num += 1
        self.sent_bytes += len(serialized_data)

    def recv(self):
        serialized_ack, addr = self.sock.recvfrom(1600)

        if addr != self.peer_addr:
            return

        ack = datagram_pb2.Ack()
        ack.ParseFromString(serialized_ack)

        action = self.action

        self.update_state(ack)

        if self.step_start_ms is None:
            self.step_start_ms = curr_ts_ms()

        done = False

        # At each step end, feed the state:
        if curr_ts_ms() - self.step_start_ms > self.step_len_ms:  # step's end
            self.state = [self.delay_ewma,
                     self.delivery_rate_ewma,
                     self.send_rate_ewma,
                     self.cwnd]
            #print(state)

            # time how long it takes to get an action from the NN
            if self.debug:
                start_sample = time.time()

            norm_state = normalize(self.state)
            one_hot_action = one_hot(self.action, self.action_cnt)
            state = norm_state + one_hot_action

            #self.action = self.sample_action(state)
            self.action = 0

            if self.debug:
                self.sampling_file.write('%.2f ms\n' % ((time.time() - start_sample) * 1000))

            self.take_action(self.action)

            '''
            self.delay_ewma = None
            self.delivery_rate_ewma = None
            self.send_rate_ewma = None
            '''

            self.step_start_ms = curr_ts_ms()

            done = False
            self.step_cnt += 1
            tput, perc_delay,reward = self.compute_performance()
            if self.step_cnt >= Sender.max_steps:
                self.step_cnt = 0
                self.running = False
                done = True
            #print self.state,self.action, reward, done

            return  True, self.state, action, reward, done, tput, perc_delay
        else:
            return False, None, None, None, None, None, None


    def compute_performance(self):
        duration = curr_ts_ms() - self.ts_first
        tput = 0.008 * self.delivered / duration
        #Compute the qth percentile of the data along the specified axis.
        perc_delay = np.percentile(self.rtt_buf, 95)
        util = self.utility
        self.utility = tput - 0.1*perc_delay


        with open(path.join(project_root.DIR, 'env', 'perf'), 'a', 0) as perf:
            perf.write('%.2f %d\n' % (tput, perc_delay))

        #print(self.utility)
        if self.utility - util  > 0.0001:
            reward = 1

        elif self.utility - util  < -0.0001:
            reward = -1
        else:
            reward = 0
        #print(tput, perc_delay, self.utility, reward)
        return tput, perc_delay,reward


    def run(self):
        TIMEOUT = 1000

        self.poller.modify(self.sock, ALL_FLAGS)
        curr_flags = ALL_FLAGS
        tput_list = []
        delay_list = []
        steps = 1


        while self.running:
            if self.window_is_open():
                if curr_flags != ALL_FLAGS:
                    self.poller.modify(self.sock, ALL_FLAGS)
                    curr_flags = ALL_FLAGS
            else:
                if curr_flags != READ_ERR_FLAGS:
                    self.poller.modify(self.sock, READ_ERR_FLAGS)
                    curr_flags = READ_ERR_FLAGS

            #Polls the set of registered file descriptors, and returns a possibly-empty list containing (fd, event) 2-tuples for the descriptors that have events or errors to report.
            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                self.send()

            for fd, flag in events:
                #fileno():Return the socket's file descriptor (a small integer)
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Error occurred to the channel')

                if flag & READ_FLAGS:

                    s0 = self.state
                    norm_state = normalize(s0)
                    one_hot_action = one_hot(self.action, self.action_cnt)
                    s0 = norm_state + one_hot_action

                    step_end ,s1, action, reward, done,tput, perc_delay = self.recv()

                    if step_end:
                        norm_state = normalize(s1)
                        one_hot_action = one_hot(self.action, self.action_cnt)
                        s1 = norm_state + one_hot_action

                        buffer.add([[s0,action,reward,s1,done]])
                        if steps > 500 and steps % 4 == 0:
                            self.update_Qnet(buffer)

                        rList.append(reward)
                        tput_list.append(tput)
                        delay_list.append(perc_delay)

                        #print(reward)

                        if steps % 1000 == 0:
                            r_ave.append(sum(rList)/1000.0)
                            print("average reward on last 1000 steps", r_ave[-1])
                            print("average tput on last 1000 steps", sum(tput_list[-1000:])/1000.0)
                            print("average delay on last 1000 steps", sum(delay_list[-1000:])/1000.0)
                            rList = []

                        steps += 1

                if flag & WRITE_FLAGS:
                    if self.window_is_open():
                        self.send()

