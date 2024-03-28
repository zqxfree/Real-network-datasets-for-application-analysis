import numpy as np
from scapy.all import rdpcap
import os
from scipy.stats import median_abs_deviation
def read_pcap_file(pcap_file):
    packets = rdpcap(pcap_file)
    sessions = []
    for packet in packets:
        if packet.haslayer('IP'):
            src_ip = packet['IP'].src
            if src_ip[:6] == '100.64':
                sessions.append(-len(packet))
            else:
                sessions.append(len(packet))
    return packets, sessions
def preprocess_sessions(session, preprocess_len=48):
    return session[:preprocess_len]
def feature_wash(stream):
    pkt_count = len(stream)
    stream = np.array(stream)
    stream[:, 0] = np.r_[0, np.diff(np.abs(stream[:, 0])).astype(float)]
    stream = stream.astype(float)
    stream = np.vstack([np.percentile(stream, np.linspace(0., 100., 5), 0), np.mean(stream, 0), np.std(stream, 0), median_abs_deviation(stream, 0)])
    return stream, pkt_count

# define file directories
session_parent_dir = r'.\datasets\BRAS-origin-datasets' # r'.\datasets\ONU-origin-datasets'
session_dir = os.listdir(session_parent_dir)
label_dict = dict(zip(['network-storage', 'network-transmission', 'video', 'game', 'instant-message', 'web-browsing', 'mail-service'], list(range(7))))
labels = []
sessions = []
flow_features = []

# extract traffic feature
for pcap_file in session_dir:
    packets, session = read_pcap_file(session_parent_dir + '/' + pcap_file)
    upstream, downstream, bistream = [], [], []
    upstream_port, downstream_port = [], []
    flow_proto = 2 # 0:TCP,1:UDP,2:other
    for pkt in packets:
        if pkt['IP'].src[:6] == '100.64':
            upstream_port.extend([pkt.sport, pkt.dport])
            upstream.append([pkt.time , pkt.len])
            bistream.append([pkt.time , pkt.len])
        if pkt['IP'].dst[:6] == '100.64':
            downstream_port.extend([pkt.sport, pkt.dport])
            downstream.append([pkt.time , pkt.len])
            bistream.append([pkt.time , pkt.len])
        if pkt.payload.name == 'IP':
            if pkt['IP'].payload.name == 'TCP':
                flow_proto = 0
            elif pkt['IP'].payload.name == 'UDP':
                flow_proto = 1
    if len(bistream) == 0:
        continue
    bistream = feature_wash(bistream)[0]
    if len(upstream) > 0:
        upstream, up_pkt_count = feature_wash(upstream)
    else:
        upstream = np.zeros((8, 2))
        up_pkt_count = 0
    if len(downstream) > 0:
        downstream, down_pkt_count = feature_wash(downstream)
    else:
        downstream = np.zeros((8, 2))
        down_pkt_count = 0
    upstream_port = np.array([upstream_port.count(443), upstream_port.count(80), upstream_port.count(8080), upstream_port.count(8000)])
    downstream_port = np.array([downstream_port.count(443), downstream_port.count(80), downstream_port.count(8080), downstream_port.count(8000)])
    labels.append(label_dict[pcap_file.split('_')[0]])
    session = preprocess_sessions(session, preprocess_len=48)
    sessions.append(session)
    flow_features.append(np.r_[np.hstack([bistream, upstream, downstream]).ravel(), flow_proto, up_pkt_count, upstream_port, down_pkt_count, upstream_port + downstream_port])
flow_features = np.vstack(flow_features)

# normalization
flow_features = (flow_features - flow_features.min(0)) / (flow_features.max(0) - flow_features.min(0))