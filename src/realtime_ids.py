"""
Experimental live packet monitor.
It aggregates packets into bidirectional 5-tuples and predicts when a flow closes
or reaches the idle timeout. Packet-derived features are approximations of NSL-KDD
connection features; this is a demonstration, not production-grade detection.
"""
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import argparse
import time
import joblib
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, ICMP

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ids_random_forest.joblib"
MODEL = None
FLOWS = {}
RECENT_DEST_CONNECTIONS = defaultdict(deque)
IDLE_TIMEOUT = 10.0
COUNT_WINDOW = 2.0

def protocol_name(packet):
    if TCP in packet: return "tcp"
    if UDP in packet: return "udp"
    if ICMP in packet: return "icmp"
    return "other"

def flow_key(packet):
    ip = packet[IP]
    proto = protocol_name(packet)
    if TCP in packet:
        sport, dport = int(packet[TCP].sport), int(packet[TCP].dport)
    elif UDP in packet:
        sport, dport = int(packet[UDP].sport), int(packet[UDP].dport)
    else:
        sport, dport = 0, 0
    left = (ip.src, sport)
    right = (ip.dst, dport)
    if left <= right:
        return (ip.src, ip.dst, proto, sport, dport), True
    return (ip.dst, ip.src, proto, dport, sport), False

def predict_flow(key, flow):
    # NSL-KDD count is a connection-context feature. This approximation counts
    # recently observed flows to the destination host.
    now = time.time()
    dest = flow['destination']
    recent = RECENT_DEST_CONNECTIONS[dest]
    while recent and now - recent[0] > COUNT_WINDOW:
        recent.popleft()
    recent.append(now)
    row = pd.DataFrame([{
        'protocol_type': flow['protocol'],
        'src_bytes': flow['src_bytes'],
        'dst_bytes': flow['dst_bytes'],
        'count': len(recent)
    }], columns=['protocol_type', 'src_bytes', 'dst_bytes', 'count'])
    result = int(MODEL.predict(row)[0])
    label = "ATTACK / SUSPICIOUS" if result == 1 else "NORMAL"
    print(f"[{datetime.now():%H:%M:%S}] {label} | "
          f"{flow['source']} -> {flow['destination']} | "
          f"{flow['protocol']} | bytes={flow['src_bytes']}/{flow['dst_bytes']}")

def handle_packet(packet):
    if IP not in packet:
        return
    now = time.time()
    key, forward = flow_key(packet)
    ip = packet[IP]
    if key not in FLOWS:
        FLOWS[key] = {
            'source': ip.src if forward else ip.dst,
            'destination': ip.dst if forward else ip.src,
            'protocol': protocol_name(packet),
            'src_bytes': 0, 'dst_bytes': 0, 'last_seen': now
        }
    flow = FLOWS[key]
    if forward:
        flow['src_bytes'] += len(packet)
    else:
        flow['dst_bytes'] += len(packet)
    flow['last_seen'] = now

    close = (TCP in packet and (packet[TCP].flags & 0x05) != 0)
    if close:
        predict_flow(key, flow)
        FLOWS.pop(key, None)

def flush_idle():
    now = time.time()
    expired = [key for key, value in FLOWS.items()
               if now - value['last_seen'] >= IDLE_TIMEOUT]
    for key in expired:
        flow = FLOWS.pop(key)
        predict_flow(key, flow)

def main():
    global MODEL
    parser = argparse.ArgumentParser(description="Experimental Scapy-based IDS monitor")
    parser.add_argument('--iface', help='Network interface name (default: Scapy default)')
    parser.add_argument('--idle-timeout', type=float, default=IDLE_TIMEOUT)
    args = parser.parse_args()
    MODEL = joblib.load(MODEL_PATH)
    print("Experimental IDS packet monitor started. Press Ctrl+C to stop.")
    print("Run only on networks you own or are authorized to monitor.")
    try:
        while True:
            sniff(iface=args.iface, prn=handle_packet, store=False, timeout=1)
            flush_idle()
    except KeyboardInterrupt:
        print("\\nStopping monitor; classifying remaining flows...")
        for key, flow in list(FLOWS.items()):
            predict_flow(key, flow)
        FLOWS.clear()

if __name__ == '__main__':
    main()
