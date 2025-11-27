
# TSHARK_PATH = r"C:\Program Files\Wireshark\tshark.exe"  # Path to your tshark.exe
# INTERFACE = "2"                                        # Interface ID from tshark -D
# CAPTURE_FILTER = "ip"                                  # Capture filter (e.g., "ip", "tcp or udp")
# KAFKA_BOOTSTRAP = "localhost:9092"                     # Kafka server
# TOPIC = "network-flows"                                # Kafka topic name



# live_tshark_to_kafka.py
# import subprocess, csv, json, time, threading, sys
# from confluent_kafka import Producer

# # ======= CONFIG - edit these =======
# TSHARK_PATH = r"C:\Program Files\Wireshark\tshark.exe"
# INTERFACE = "5"       # Your interface ID from tshark -D
# CAPTURE_FILTER = "ip" # Recommended filter
# KAFKA_BOOTSTRAP = "localhost:9092"
# TOPIC = "network-flows"
# FLOW_TIMEOUT = 5.0
# # ===================================

# # build tshark command
# base_cmd = [
#     TSHARK_PATH,
#     "-i", INTERFACE,
#     "-T", "fields",
#     "-E", "header=y", "-E", "separator=,",
#     "-e", "frame.time_epoch",
#     "-e", "ip.src", "-e", "ip.dst",
#     "-e", "tcp.srcport", "-e", "tcp.dstport",
#     "-e", "udp.srcport", "-e", "udp.dstport",
#     "-e", "_ws.col.Protocol",
#     "-e", "frame.len",
#     "-l"
# ]
# if CAPTURE_FILTER:
#     base_cmd += ["-f", CAPTURE_FILTER]

# p = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
# flows = {}

# def emit_flow(key, f):
#     msg = {
#         "src_ip": key[0], "dst_ip": key[1],
#         "src_port": key[2], "dst_port": key[3],
#         "protocol": key[4],
#         "pkt_count": f["pkt_count"],
#         "byte_count": f["byte_count"],
#         "start_time": f["start_time"],
#         "last_time": f["last_time"],
#         "duration": round(f["last_time"] - f["start_time"], 6)
#     }
#     p.produce(TOPIC, json.dumps(msg).encode("utf-8"))
#     p.poll(0)

# def flow_flusher():
#     while True:
#         now = time.time()
#         to_emit = []
#         for k,f in list(flows.items()):
#             if now - f["last_time"] > FLOW_TIMEOUT:
#                 to_emit.append((k,f))
#                 del flows[k]
#         for k,f in to_emit:
#             emit_flow(k,f)
#         time.sleep(1.0)

# threading.Thread(target=flow_flusher, daemon=True).start()
# proc = subprocess.Popen(base_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
# reader = csv.DictReader(proc.stdout)

# try:
#     for row in reader:
#         try:
#             ts = float(row.get("frame.time_epoch") or time.time())
#             src = row.get("ip.src") or ""
#             dst = row.get("ip.dst") or ""
#             proto = row.get("_ws.col.Protocol") or ""
#             sport = row.get("tcp.srcport") or row.get("udp.srcport") or ""
#             dport = row.get("tcp.dstport") or row.get("udp.dstport") or ""
#             length = int(row.get("frame.len") or 0)
#         except Exception:
#             continue
#         key = (src, dst, sport, dport, proto)
#         f = flows.get(key)
#         if not f:
#             flows[key] = {"pkt_count": 1, "byte_count": length, "start_time": ts, "last_time": ts}
#         else:
#             f["pkt_count"] += 1
#             f["byte_count"] += length
#             f["last_time"] = ts
# except KeyboardInterrupt:
#     for k,f in list(flows.items()):
#         emit_flow(k,f)
#     p.flush(10)
#     proc.terminate()
#     sys.exit(0)



# --------------
# import subprocess
# import csv
# import json
# import time
# import threading
# import sys
# from confluent_kafka import Producer

# # ===== CONFIG =====
# TSHARK_PATH = r"C:\Program Files\Wireshark\tshark.exe"  # Path to tshark
# INTERFACE = "5"        # Replace with your interface ID from `tshark -D`
# CAPTURE_FILTER = "ip"  # Recommended: "ip" or "tcp or udp"
# KAFKA_BOOTSTRAP = "localhost:9092"
# TOPIC = "test"
# FLOW_TIMEOUT = 5.0     # seconds to wait before finalizing a flow
# # ==================

# # Kafka producer
# producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

# # Flow state table
# flows = {}

# # Build tshark command
# base_cmd = [
#     TSHARK_PATH,
#     "-i", INTERFACE,
#     "-T", "fields",
#     "-E", "header=y", "-E", "separator=,",
#     "-e", "frame.time_epoch",
#     "-e", "ip.src", "-e", "ip.dst",
#     "-e", "tcp.srcport", "-e", "tcp.dstport",
#     "-e", "udp.srcport", "-e", "udp.dstport",
#     "-e", "_ws.col.Protocol",
#     "-e", "frame.len",
#     "-l"  # line-buffered
# ]
# if CAPTURE_FILTER:
#     base_cmd += ["-f", CAPTURE_FILTER]

# def emit_flow(key, f):
#     """Send a completed flow as JSON to Kafka."""
#     msg = {
#         "src_ip": key[0],
#         "dst_ip": key[1],
#         "src_port": key[2],
#         "dst_port": key[3],
#         "protocol": key[4],
#         "pkt_count": f["pkt_count"],
#         "byte_count": f["byte_count"],
#         "start_time": f["start_time"],
#         "last_time": f["last_time"],
#         "duration": round(f["last_time"] - f["start_time"], 6)
#     }
#     try:
#         producer.produce(TOPIC, json.dumps(msg).encode("utf-8"))
#         producer.poll(0)
#     except Exception as e:
#         print(f"Error producing to Kafka: {e}")

# def flow_flusher():
#     """Background thread that flushes expired flows to Kafka."""
#     while True:
#         now = time.time()
#         to_emit = []
#         for k, f in list(flows.items()):
#             if now - f["last_time"] > FLOW_TIMEOUT:
#                 to_emit.append((k, f))
#                 del flows[k]
#         for k, f in to_emit:
#             emit_flow(k, f)
#         time.sleep(1.0)

# # Start background flusher thread
# threading.Thread(target=flow_flusher, daemon=True).start()

# # Start tshark process
# proc = subprocess.Popen(
#     base_cmd,
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
#     text=True,
#     bufsize=1
# )
# reader = csv.DictReader(proc.stdout)

# try:
#     for row in reader:
#         try:
#             ts = float(row.get("frame.time_epoch") or time.time())
#             src = row.get("ip.src") or ""
#             dst = row.get("ip.dst") or ""
#             proto = row.get("_ws.col.Protocol") or ""
#             sport = row.get("tcp.srcport") or row.get("udp.srcport") or "0"
#             dport = row.get("tcp.dstport") or row.get("udp.dstport") or "0"
#             length = int(row.get("frame.len") or 0)
#         except Exception:
#             continue

#         key = (src, dst, sport, dport, proto)
#         f = flows.get(key)
#         if not f:
#             flows[key] = {
#                 "pkt_count": 1,
#                 "byte_count": length,
#                 "start_time": ts,
#                 "last_time": ts
#             }
#         else:
#             f["pkt_count"] += 1
#             f["byte_count"] += length
#             f["last_time"] = ts

# except KeyboardInterrupt:
#     print("\nStopping capture...")
#     for k, f in list(flows.items()):
#         emit_flow(k, f)
#     producer.flush(10)
#     proc.terminate()
#     sys.exit(0)




# import subprocess
# import csv
# import json
# import time
# import threading
# import sys
# from confluent_kafka import Producer

# # ===== CONFIG =====
# TSHARK_PATH = r"C:\Program Files\Wireshark\tshark.exe"  # Correct tshark path
# INTERFACE = "4"       # Wi-Fi interface (you found from tshark -D)
# CAPTURE_FILTER = "ip"  # Recommended filter
# KAFKA_BOOTSTRAP = "localhost:9092"
# TOPIC = "test"
# FLOW_TIMEOUT = 5.0     # seconds
# # ==================

# # Kafka producer
# producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

# # Active flows
# flows = {}

# # Build tshark command
# base_cmd = [
#     TSHARK_PATH,
#     "-i", INTERFACE,
#     "-T", "fields",
#     "-E", "header=y",
#     "-E", "separator=,",
#     "-e", "frame.time_epoch",
#     "-e", "ip.src",
#     "-e", "ip.dst",
#     "-e", "tcp.srcport",
#     "-e", "tcp.dstport",
#     "-e", "udp.srcport",
#     "-e", "udp.dstport",
#     "-e", "_ws.col.Protocol",
#     "-e", "frame.len",
#     "-l"
# ]

# if CAPTURE_FILTER:
#     base_cmd += ["-f", CAPTURE_FILTER]


# def emit_flow(key, f):
#     """Send completed flow to Kafka"""
#     msg = {
#         "src_ip": key[0],
#         "dst_ip": key[1],
#         "src_port": key[2],
#         "dst_port": key[3],
#         "protocol": key[4],
#         "pkt_count": f["pkt_count"],
#         "byte_count": f["byte_count"],
#         "start_time": f["start_time"],
#         "last_time": f["last_time"],
#         "duration": round(f["last_time"] - f["start_time"], 6)
#     }
#     try:
#         producer.produce(TOPIC, json.dumps(msg).encode("utf-8"))
#         producer.poll(0)
#     except Exception as e:
#         print("Kafka Error:", e)


# def flow_flusher():
#     """Flush expired flows to Kafka"""
#     while True:
#         now = time.time()
#         expired = []
#         for key, f in list(flows.items()):
#             if now - f["last_time"] > FLOW_TIMEOUT:
#                 expired.append((key, f))
#                 del flows[key]

#         for key, f in expired:
#             emit_flow(key, f)

#         time.sleep(1)


# # Start background flusher
# threading.Thread(target=flow_flusher, daemon=True).start()

# # Start tshark process
# proc = subprocess.Popen(
#     base_cmd,
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
#     text=True,
#     bufsize=1
# )

# reader = csv.DictReader(proc.stdout)

# print("🔥 Capturing live packets... (Press Ctrl+C to stop)")
# print("🔥 Sending flow records to Kafka topic:", TOPIC)

# try:
#     for row in reader:
#         try:
#             ts = float(row.get("frame.time_epoch") or time.time())
#             src = row.get("ip.src") or ""
#             dst = row.get("ip.dst") or ""
#             proto = row.get("_ws.col.Protocol") or ""
#             sport = row.get("tcp.srcport") or row.get("udp.srcport") or "0"
#             dport = row.get("tcp.dstport") or row.get("udp.dstport") or "0"
#             length = int(row.get("frame.len") or 0)
#         except:
#             continue

#         key = (src, dst, sport, dport, proto)

#         if key not in flows:
#             flows[key] = {
#                 "pkt_count": 1,
#                 "byte_count": length,
#                 "start_time": ts,
#                 "last_time": ts
#             }
#         else:
#             f = flows[key]
#             f["pkt_count"] += 1
#             f["byte_count"] += length
#             f["last_time"] = ts

# except KeyboardInterrupt:
#     print("\nStopping packet capture...")
#     for key, f in list(flows.items()):
#         emit_flow(key, f)
#     producer.flush(10)
#     proc.terminate()
#     sys.exit(0)




import subprocess
import csv
import json
import time
import threading
import sys
from confluent_kafka import Producer

# ===== CONFIG =====
TSHARK_PATH = r"C:\Program Files\Wireshark\tshark.exe"
INTERFACE = "4"            # Wi-Fi interface (correct for your PC)
CAPTURE_FILTER = ""        # keep empty to capture ALL packets
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "test"
FLOW_TIMEOUT = 5.0
# ==================

producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
flows = {}

# Tshark command
base_cmd = [
    TSHARK_PATH,
    "-i", INTERFACE,
    "-T", "fields",
    "-E", "header=y", "-E", "separator=,",
    "-e", "frame.time_epoch",
    "-e", "ip.src", "-e", "ip.dst",
    "-e", "tcp.srcport", "-e", "tcp.dstport",
    "-e", "udp.srcport", "-e", "udp.dstport",
    "-e", "_ws.col.Protocol",
    "-e", "frame.len",
    "-l"
]

if CAPTURE_FILTER:
    base_cmd += ["-f", CAPTURE_FILTER]

def emit_flow(key, f):
    msg = {
        "src_ip": key[0],
        "dst_ip": key[1],
        "src_port": key[2],
        "dst_port": key[3],
        "protocol": key[4],
        "pkt_count": f["pkt_count"],
        "byte_count": f["byte_count"],
        "start_time": f["start_time"],
        "last_time": f["last_time"],
        "duration": round(f["last_time"] - f["start_time"], 6)
    }

    producer.produce(TOPIC, json.dumps(msg).encode("utf-8"))
    producer.poll(0)

def flow_flusher():
    while True:
        now = time.time()
        expired = []

        for k, f in list(flows.items()):
            if now - f["last_time"] > FLOW_TIMEOUT:
                expired.append((k, f))
                del flows[k]

        for k, f in expired:
            emit_flow(k, f)

        time.sleep(1)

# Start background thread
threading.Thread(target=flow_flusher, daemon=True).start()

print("🔥 Capturing live packets... (Ctrl+C to stop)")
print("🔥 Using interface:", INTERFACE)
print("🔥 Sending to Kafka topic:", TOPIC)

proc = subprocess.Popen(
    base_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

reader = csv.DictReader(proc.stdout)

# Print tshark errors (important)
def read_errors():
    for line in proc.stderr:
        print("TSHARK ERROR:", line.strip())

threading.Thread(target=read_errors, daemon=True).start()

try:
    for row in reader:
        print("ROW:", row)     # <-- SEE LIVE PACKETS HERE

        ts = float(row.get("frame.time_epoch") or time.time())
        src = row.get("ip.src") or ""
        dst = row.get("ip.dst") or ""
        proto = row.get("_ws.col.Protocol") or ""
        sport = row.get("tcp.srcport") or row.get("udp.srcport") or "0"
        dport = row.get("tcp.dstport") or row.get("udp.dstport") or "0"
        length = int(row.get("frame.len") or 0)

        key = (src, dst, sport, dport, proto)
        f = flows.get(key)

        if not f:
            flows[key] = {
                "pkt_count": 1,
                "byte_count": length,
                "start_time": ts,
                "last_time": ts
            }
        else:
            f["pkt_count"] += 1
            f["byte_count"] += length
            f["last_time"] = ts

except KeyboardInterrupt:
    print("\nStopping capture...")
    for k, f in list(flows.items()):
        emit_flow(k, f)
    producer.flush(10)
    proc.terminate()
    sys.exit(0)
