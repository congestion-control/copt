#adjusted from https://github.com/StanfordSNR/indigo/blob/master/env/receiver.py

import sys
import json
import socket
import select
import datagram_pb2
import project_root
from helpers.helpers import READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, ALL_FLAGS

class Receiver(object):
    def __init__(self,ip,port):
        self.peer_addr = (ip, port)

        #UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

    def cleanup(self):
        self.sock.close()

    def construct_ack_from_data(self, serialized_data):
        """Construct a serialized ACK that acks a serialized datagram."""

        data = datagram_pb2.Data()
        data.ParseFromString(serialized_data)

        ack = datagram_pb2.Ack()
        ack.seq_num = data.seq_num
        ack.send_ts = data.send_ts
        ack.sent_bytes = data.sent_bytes
        ack.delivered_time = data.delivered_time
        ack.delivered = data.delivered
        ack.ack_bytes = len(serialized_data)

        return ack.SerializeToString()

    def handshake(self):
        # non-blocking UDP socket
        self.sock.setblocking(0)

        TIMEOUT = 1000

        retry_times = 0
        self.poller.modify(self.sock, READ_ERR_FLAGS)
        #print(self.peer_addr)

        while True:
            self.sock.sendto('Hello from receiver', self.peer_addr)
            events = self.poller.poll(TIMEOUT)

            if not events:
                retry_times += 1
                if retry_times > 10:
                    sys.stderr.write('[receiver] Handshake failed after 10 retries\n')
                    return
                else:
                    sys.stderr.write('[receiver] Handshake timed out and retrying...\n')
                    continue

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Channel closed ot error occured')

                if flag & READ_FLAGS:

                    msg, addr = self.sock.recvfrom(1600)

                    #print(addr, self.peer_addr)
                    if addr == self.peer_addr:

                        if msg != 'Hello from sender':
                            # 'Hello from sender' was presumably lost
                            # received subsequent data from peer sender
                            ack = self.construct_ack_from_data(msg)
                            if ack is not None:
                                self.sock.sendto(ack, self.peer_addr)

                        return

    def run(self):
        self.sock.setblocking(1)

        while True:
            serialized_data, addr = self.sock.recvfrom(1600)
            #print(serialized_data)
            if addr == self.peer_addr:
                ack = self.construct_ack_from_data(serialized_data)
                if ack is not None:
                    self.sock.sendto(ack, self.peer_addr)