# # backend/utils/preprocess.py
# import numpy as np
# import torch

# # Example: convert IP to numeric hash
# def ip_hash(ip):
#     try:
#         parts = [int(x) for x in str(ip).split('.') if x != '']
#         return sum(parts) % 256
#     except Exception:
#         return 0

# PROTO_MAP = {
#     "TCP": 1, "UDP": 2, "DNS": 3, "HTTPS": 4, "HTTP": 5
# }

# def flow_to_feature_vector(flow_json):
#     """
#     Converts flow JSON to feature vector:
#     [src_ip_h, dst_ip_h, src_port, dst_port, protocol_num, pkt_count, byte_count, duration]
#     """
#     src_ip = flow_json.get("src_ip", "")
#     dst_ip = flow_json.get("dst_ip", "")
#     src_port = float(flow_json.get("src_port", 0) or 0)
#     dst_port = float(flow_json.get("dst_port", 0) or 0)
#     proto = str(flow_json.get("protocol", "")).upper()
#     proto_num = PROTO_MAP.get(proto, 0)
#     pkt_count = float(flow_json.get("pkt_count", 0) or 0)
#     byte_count = float(flow_json.get("byte_count", 0) or 0)
#     duration = float(flow_json.get("duration", 0) or 0)

#     vec = [
#         float(ip_hash(src_ip)),
#         float(ip_hash(dst_ip)),
#         src_port,
#         dst_port,
#         float(proto_num),
#         pkt_count,
#         byte_count,
#         duration
#     ]
#     return np.array(vec, dtype=float)

# def make_tensor_from_vec(vec, scaler, device):
#     """
#     vec: numpy array (feat_dim,)
#     returns: tensor (1,1,feat_dim) scaled and ready for model
#     """
#     scaled = scaler.transform(vec.reshape(1, -1))
#     tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(1).to(device)
#     return tensor





# backend/utils/preprocess.py
import numpy as np
import torch

# Example: convert IP to numeric hash
def ip_hash(ip):
    try:
        # handle IPv4 dotted strings
        if isinstance(ip, (list, tuple)):
            parts = ip
        else:
            parts = [int(x) for x in str(ip).split('.') if x != '']
        return sum(parts) % 256
    except Exception:
        # fallback for non-IPv4 strings
        s = str(ip)
        h = sum(ord(c) for c in s) % 256
        return float(h)

PROTO_MAP = {
    "TCP": 1, "UDP": 2, "DNS": 3, "HTTPS": 4, "HTTP": 5
}

def flow_to_feature_vector(flow_json):
    """
    Converts flow JSON to feature vector:
    [src_ip_h, dst_ip_h, src_port, dst_port, protocol_num, pkt_count, byte_count, duration]
    Accepts protocol as string (TCP) or numeric (6).
    """
    src_ip = flow_json.get("src_ip", "")
    dst_ip = flow_json.get("dst_ip", "")

    try:
        src_port = float(flow_json.get("src_port", 0) or 0)
    except Exception:
        src_port = 0.0
    try:
        dst_port = float(flow_json.get("dst_port", 0) or 0)
    except Exception:
        dst_port = 0.0

    proto_raw = flow_json.get("protocol", "")
    # Accept numeric protocol (e.g. 6) or string (e.g. "TCP")
    if isinstance(proto_raw, (int, float)):
        proto_num = int(proto_raw)
    else:
        proto_str = str(proto_raw).upper()
        proto_num = PROTO_MAP.get(proto_str, 0)

    try:
        pkt_count = float(flow_json.get("pkt_count", 0) or 0)
    except Exception:
        pkt_count = 0.0
    try:
        byte_count = float(flow_json.get("byte_count", 0) or 0)
    except Exception:
        byte_count = 0.0
    try:
        duration = float(flow_json.get("duration", 0) or 0)
    except Exception:
        duration = 0.0

    vec = [
        float(ip_hash(src_ip)),
        float(ip_hash(dst_ip)),
        src_port,
        dst_port,
        float(proto_num),
        pkt_count,
        byte_count,
        duration
    ]
    return np.array(vec, dtype=float)

def make_tensor_from_vec(vec, scaler, device):
    """
    vec: numpy array shape (feat_dim,)
    returns: tensor (1,1,feat_dim) scaled and ready for model
    """
    vec = vec.reshape(1, -1)
    try:
        scaled = scaler.transform(vec)
    except Exception as e:
        # try to coerce shape
        import numpy as _np
        scaled = scaler.transform(_np.asarray(vec, dtype=float))
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(1).to(device)
    return tensor
