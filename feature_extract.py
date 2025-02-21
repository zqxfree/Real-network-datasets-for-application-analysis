import numpy as np
from scapy.all import rdpcap
import os
from scipy.stats import median_abs_deviation

def feature_wash(stream):
    pkt_count = len(stream)
    stream = np.array(stream)
    stream[:, 0] = np.r_[0, np.diff(np.abs(stream[:, 0])).astype(float)]
    stream = stream.astype(float)
    stream = np.vstack([np.percentile(stream, np.linspace(0., 100., 5), 0), np.mean(stream, 0), np.std(stream, 0), median_abs_deviation(stream, 0)])
    return stream, pkt_count

# extract traffic feature
def extract_feature_label(session_parent_dir):
    labels = []
    flow_features = []
    session_dir = os.listdir(session_parent_dir)
    for pcap_file in session_dir:
        packets = rdpcap(session_parent_dir + '/' + pcap_file)
        if len(packets) < 3:
            continue
        upstream, downstream, bistream = [], [], []
        upstream_port, downstream_port = [], []
        flow_proto = 2  # 0:TCP,1:UDP,2:other
        user_ip_prefix = ('100.64', '192.168')
        for pkt in packets:
            if pkt['IP'].src[:6] in user_ip_prefix:
                upstream_port.extend([pkt.sport, pkt.dport])
                upstream.append([pkt.time, pkt.len])
                bistream.append([pkt.time, pkt.len])
            else:
                downstream_port.extend([pkt.sport, pkt.dport])
                downstream.append([pkt.time, pkt.len])
                bistream.append([pkt.time, pkt.len])
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
            downstream, _ = feature_wash(downstream)
        else:
            downstream = np.zeros((8, 2))
        upstream_port = np.array([upstream_port.count(443), upstream_port.count(80)])
        downstream_port = np.array([downstream_port.count(443), downstream_port.count(80)])
        labels.append(pcap_file.split('_')[2])
        flow_features.append(np.r_[np.hstack(
            [bistream, upstream, downstream]).ravel(), up_pkt_count, upstream_port, downstream_port, flow_proto])
    flow_features = np.vstack(flow_features)
    labels = np.array(labels)
    # normalization
    flow_features = (flow_features - flow_features.min(0)) / (flow_features.max(0) - flow_features.min(0) + 1e-4)
    return flow_features, labels

if __name__ == "__main__":
    # define file directories
    flow_features, labels = extract_feature_label(r'.\dataset\session_example')
    print(flow_features.shape, labels.shape)
    print(flow_features)
    print(labels)
    # store feature and label data
    np.save(r'.\dataset\session_extracted\flow_features.npy', flow_features)
    np.save(r'.\dataset\session_extracted\labels.npy', labels)