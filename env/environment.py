#adjusted from https://github.com/StanfordSNR/indigo/blob/master/env/environment.py


import os
from os import path
import sys
import signal
from subprocess import Popen
from sender import Sender
import project_root
from helpers.helpers import get_open_udp_port


class Environment(object):
    def __init__(self, mahimahi_cmd):
        self.mahimahi_cmd = mahimahi_cmd
        self.state_dim = Sender.state_dim
        self.action_cnt = Sender.action_cnt

        self.sender = None
        self.receiver = None

    def set_sample_action(self, sample_action):
        """Set the sender's policy. Must be called before calling reset()."""
        self.sample_action = sample_action

    def set_update_Qnet(self, update_Qnet):
        """Set the sender's policy. Must be called before calling reset()."""
        self.update_Qnet = update_Qnet

    def reset(self):
        """Must be called before running rollout()."""

        self.cleanup()

        self.port = get_open_udp_port()

        #start sender as  an instance of Sender class
        sys.stderr.write('Starting sender...\n')
        self.sender = Sender(self.port, train=True)
        self.sender.set_sample_action(self.sample_action)
        self.sender.set_update_Qnet(self.update_Qnet)

        # start receiver in a subprocess
        sys.stderr.write('Starting receiver...\n')
        receiver_src = path.join(project_root.DIR, 'env', 'run_receiver.py') # path of run_receiver.py
        recv_cmd = 'python %s $MAHIMAHI_BASE %s' % (receiver_src, self.port)
        cmd = "%s -- sh -c '%s'" % (self.mahimahi_cmd, recv_cmd)
        sys.stderr.write('$ %s\n' % cmd)
        #Popen: Execute a child program in a new process.
        #preexec_fn is set to a callable object, this object will be called in the child process just before the child is executed. (Unix only)
        #os.setsid: To keep the child process running while the parent process exit.
        self.receiver = Popen(cmd, preexec_fn=os.setsid, shell=True)

        # sender completes the handshake sent from receiver
        self.sender.handshake()
        state = [self.sender.delay_ewma, self.sender.send_rate_ewma, self.sender.delivery_rate_ewma, self.sender.cwnd]
        return state

    def rollout(self):
        sys.stderr.write('Obtaining an episode from environment...\n')
        return self.sender.run()

    def cleanup(self):
        if self.sender:
            self.sender.cleanup()
            self.sender = None

        if self.receiver:
            try:
                os.killpg(os.getpgid(self.receiver.pid), signal.SIGTERM)
            except OSError as e:
                sys.stderr.write('%s ' % e)
            finally:
                self.receiver = None

