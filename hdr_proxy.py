import os
import json
import socket
import threading
import queue
import time
import numpy as np
from flask import Flask, render_template_string, request, jsonify

def log(msg):
    print(f"[HDR-PROXY] {msg}", flush=True)

CONFIG_FILE = "config.json"
DDP_HEADER_SIZE = 10
LUT_SIZE = 64

PROFILE_DEFAULTS = {
    "EXPOSURE": 1.2,
    "GAMMA": 2.0,
    "SATURATION": 1.1,
    "BLACK_CUTOFF": 5,
    "SMOOTHING": 0.0,
    "GAIN_R": 1.0,
    "GAIN_G": 1.0,
    "GAIN_B": 1.0,
}

SDR_PROFILE_DEFAULTS = {**PROFILE_DEFAULTS, "INPUT_COLOR_SPACE": "REC709"}
HDR_PROFILE_DEFAULTS = {**PROFILE_DEFAULTS, "INPUT_COLOR_SPACE": "REC2020"}

DEFAULT_CONFIG = {
    "PROXY_ACTIVE": True,
    "ACTIVE_MODE": "HDR",
    "PERF_MONITOR_ACTIVE": True,
    "NUM_LEDS": 227,
    "LISTEN_IP": "0.0.0.0",
    "LISTEN_PORT": 21324,
    "WLED_IP": "192.168.100.190",
    "WLED_PORT": 4048,
    "PROFILE_SDR": SDR_PROFILE_DEFAULTS.copy(),
    "PROFILE_HDR": HDR_PROFILE_DEFAULTS.copy(),
}

config_lock = threading.RLock()
stats_lock = threading.Lock()

stats_data = {
    "received_fps": 0,
    "processed_fps": 0,
    "latency_ms": 0.0
}

def merge_dicts(base, override):
    merged = base.copy()
    if isinstance(override, dict):
        merged.update(override)
    return merged

def normalize_config(raw_config):
    if not isinstance(raw_config, dict):
        raw_config = {}

    normalized = DEFAULT_CONFIG.copy()
    normalized.update(raw_config)

    normalized["PROFILE_SDR"] = merge_dicts(SDR_PROFILE_DEFAULTS, raw_config.get("PROFILE_SDR", {}))
    normalized["PROFILE_HDR"] = merge_dicts(HDR_PROFILE_DEFAULTS, raw_config.get("PROFILE_HDR", {}))

    hdr_space = str(normalized["PROFILE_HDR"].get("INPUT_COLOR_SPACE", "REC2020")).upper()
    if hdr_space not in {"REC2020", "DCIP3"}:
        normalized["PROFILE_HDR"]["INPUT_COLOR_SPACE"] = "REC2020"

    if normalized.get("ACTIVE_MODE") not in {"SDR", "HDR"}:
        normalized["ACTIVE_MODE"] = "HDR"

    return normalized

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    return normalize_config(data)
        except Exception as e:
            log(f"Error reading config.json: {e}")
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG.copy()

def save_config(config_data):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        log(f"Error saving config.json: {e}")

config = load_config()

XYZ_TO_REC709 = np.array([
    [3.2409699419, -1.5373831776, -0.4986107603],
    [-0.9692436363, 1.8759675015, 0.0415550574],
    [0.0556300797, -0.2039769589, 1.0569715140],
], dtype=np.float32)

REC2020_TO_XYZ = np.array([
    [0.6369580483, 0.1446169036, 0.1688809752],
    [0.2627002120, 0.6779980715, 0.0593017165],
    [0.0000000000, 0.0280726930, 1.0609850577],
], dtype=np.float32)

DCI_P3_TO_XYZ = np.array([
    [0.445169815, 0.277134409, 0.172282570],
    [0.209491677, 0.721595234, 0.068913089],
    [0.000000000, 0.047060, 0.907355],
], dtype=np.float32)

def get_input_matrix(input_space):
    input_space = str(input_space).upper()
    if input_space == "REC2020":
        return np.matmul(XYZ_TO_REC709, REC2020_TO_XYZ).astype(np.float32)
    if input_space == "DCIP3":
        return np.matmul(XYZ_TO_REC709, DCI_P3_TO_XYZ).astype(np.float32)
    return np.eye(3, dtype=np.float32)

app = Flask(__name__)

LUT = np.zeros((LUT_SIZE, LUT_SIZE, LUT_SIZE, 3), dtype=np.uint8)
lut_lock = threading.Lock()
lut_queue = queue.Queue()

prev_frame_float = None
lut_updated_flag = False

def get_active_profile():
    with config_lock:
        active_mode = str(config.get("ACTIVE_MODE", "HDR")).upper()
        if active_mode == "SDR":
            profile = config.get("PROFILE_SDR", SDR_PROFILE_DEFAULTS.copy())
            input_space = "REC709"
        else:
            profile = config.get("PROFILE_HDR", HDR_PROFILE_DEFAULTS.copy())
            input_space = profile.get("INPUT_COLOR_SPACE", "REC2020")
        return {
            "mode": active_mode,
            "profile": profile,
            "input_space": input_space,
        }

def generate_lut_matrix(exposure, gamma, sat, cutoff, gain_r, gain_g, gain_b, input_space):
    steps = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float32)
    grid_r, grid_g, grid_b = np.meshgrid(steps, steps, steps, indexing='ij')
    rgb_grid = np.stack([grid_r, grid_g, grid_b], axis=-1)

    rgb_in_linear = np.power(rgb_grid, 2.2, dtype=np.float32)

    matrix = get_input_matrix(input_space)
    rgb_linear = np.matmul(rgb_in_linear, matrix.T)

    min_val = np.min(rgb_linear, axis=-1, keepdims=True)
    rgb_linear = np.where(min_val < 0.0, rgb_linear - min_val, rgb_linear)

    if str(input_space).upper() != "REC709":
        max_val = np.max(rgb_linear, axis=-1, keepdims=True)
        max_val = np.maximum(max_val, 1e-6, dtype=np.float32)

        max_scaled = max_val * exposure
        max_mapped = max_scaled / (max_scaled + 0.5)

        scale = max_mapped / max_val
        rgb_mapped = rgb_linear * scale
    else:
        rgb_mapped = np.clip(rgb_linear * exposure, 0.0, 1.0)

    lum = 0.2126 * rgb_mapped[..., 0] + 0.7152 * rgb_mapped[..., 1] + 0.0722 * rgb_mapped[..., 2]
    lum_ext = np.expand_dims(lum, axis=-1)

    rgb_mapped = lum_ext + sat * (rgb_mapped - lum_ext)
    rgb_mapped = np.clip(rgb_mapped, 0.0, 1.0)

    inv_gamma = 1.0 / max(gamma, 0.1)
    rgb_srgb = np.power(rgb_mapped, inv_gamma, dtype=np.float32)

    rgb_srgb[..., 0] *= gain_r
    rgb_srgb[..., 1] *= gain_g
    rgb_srgb[..., 2] *= gain_b

    rgb_out = rgb_srgb * 255.0
    rgb_out[rgb_out < cutoff] = 0.0
    return np.clip(rgb_out, 0.0, 255.0).astype(np.uint8)

def lut_worker():
    global LUT, lut_updated_flag
    while True:
        try:
            params = lut_queue.get()
            if params is None:
                break

            while not lut_queue.empty():
                try:
                    params = lut_queue.get_nowait()
                except queue.Empty:
                    break

            new_lut = generate_lut_matrix(*params)
            with lut_lock:
                LUT = new_lut
                lut_updated_flag = True
            log("3D LUT rebuilt successfully.")
            lut_queue.task_done()
        except Exception as e:
            log(f"Error in LUT worker: {e}")

threading.Thread(target=lut_worker, daemon=True).start()

def rebuild_lut_for_current_profile():
    profile = get_active_profile()
    active_profile = profile["profile"]
    lut_params = (
        float(active_profile.get("EXPOSURE", 1.2)),
        float(active_profile.get("GAMMA", 2.0)),
        float(active_profile.get("SATURATION", 1.1)),
        int(active_profile.get("BLACK_CUTOFF", 5)),
        float(active_profile.get("GAIN_R", 1.0)),
        float(active_profile.get("GAIN_G", 1.0)),
        float(active_profile.get("GAIN_B", 1.0)),
        profile["input_space"],
    )
    lut_queue.put(lut_params)

rebuild_lut_for_current_profile()

def extract_ddp_rgb(raw_bytes, num_leds):
    if len(raw_bytes) <= DDP_HEADER_SIZE:
        return None

    payload = raw_bytes[DDP_HEADER_SIZE:DDP_HEADER_SIZE + num_leds * 3]
    if len(payload) == 0:
        return None

    if len(payload) < num_leds * 3:
        payload = payload[: len(payload) - (len(payload) % 3)]

    rgb = np.frombuffer(payload, dtype=np.uint8)
    if rgb.size == 0:
        return None
    return rgb.reshape(-1, 3)

def tone_map_udp_fast(raw_bytes, num_leds, smoothing):
    global prev_frame_float, lut_updated_flag

    rgb_in = extract_ddp_rgb(raw_bytes, num_leds)
    if rgb_in is None:
        return raw_bytes

    idx = rgb_in >> 2
    with lut_lock:
        rgb_out = LUT[idx[:, 0], idx[:, 1], idx[:, 2]].copy()
        if lut_updated_flag:
            prev_frame_float = None
            lut_updated_flag = False

    if smoothing > 0.0:
        current_float = rgb_out.astype(np.float32)
        if prev_frame_float is None or prev_frame_float.shape != current_float.shape:
            prev_frame_float = current_float
        else:
            prev_frame_float = (prev_frame_float * smoothing) + (current_float * (1.0 - smoothing))
            rgb_out = np.clip(prev_frame_float, 0.0, 255.0).astype(np.uint8)
    else:
        prev_frame_float = None

    header = raw_bytes[:DDP_HEADER_SIZE]
    return header + rgb_out.tobytes()

def udp_loop():
    sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_in.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

    with config_lock:
        listen_ip = config["LISTEN_IP"]
        listen_port = config["LISTEN_PORT"]

    sock_in.bind((listen_ip, listen_port))
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    last_stat_time = time.perf_counter()
    recv_count = 0
    proc_count = 0
    latency_accum_ms = 0.0

    while True:
        try:
            with config_lock:
                wled_target_ip = config["WLED_IP"]
                wled_target_port = config["WLED_PORT"]
                proxy_active = config["PROXY_ACTIVE"]
                perf_monitor_active = config.get("PERF_MONITOR_ACTIVE", True)
                num_leds = config["NUM_LEDS"]
                active_mode = str(config.get("ACTIVE_MODE", "HDR")).upper()
                if active_mode == "SDR":
                    smoothing = float(config["PROFILE_SDR"].get("SMOOTHING", 0.0))
                else:
                    smoothing = float(config["PROFILE_HDR"].get("SMOOTHING", 0.0))

            data, _ = sock_in.recvfrom(4096)
            if perf_monitor_active:
                recv_count += 1

            sock_in.setblocking(False)
            while True:
                try:
                    more_data, _ = sock_in.recvfrom(4096)
                    if len(more_data) > DDP_HEADER_SIZE:
                        data = more_data
                        if perf_monitor_active:
                            recv_count += 1
                except BlockingIOError:
                    break
            sock_in.setblocking(True)

            if len(data) > DDP_HEADER_SIZE:
                t0 = time.perf_counter() if perf_monitor_active else 0
                if proxy_active:
                    processed = tone_map_udp_fast(data, num_leds, smoothing)
                    sock_out.sendto(processed, (wled_target_ip, wled_target_port))
                else:
                    sock_out.sendto(data, (wled_target_ip, wled_target_port))

                if perf_monitor_active:
                    t1 = time.perf_counter()
                    proc_count += 1
                    latency_accum_ms += (t1 - t0) * 1000.0

            if perf_monitor_active:
                now = time.perf_counter()
                elapsed = now - last_stat_time
                if elapsed >= 1.0:
                    with stats_lock:
                        stats_data["received_fps"] = int(round(recv_count / elapsed))
                        stats_data["processed_fps"] = int(round(proc_count / elapsed))
                        stats_data["latency_ms"] = round(latency_accum_ms / max(proc_count, 1), 2)

                    last_stat_time = now
                    recv_count = 0
                    proc_count = 0
                    latency_accum_ms = 0.0
            else:
                with stats_lock:
                    stats_data["received_fps"] = 0
                    stats_data["processed_fps"] = 0
                    stats_data["latency_ms"] = 0.0

        except Exception as e:
            log(f"Error in UDP socket: {e}")

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NCA2NCI+PHJlY3Qgd2lkdGg9IjY0IiBoZWlnaHQ9IjY0IiByeD0iMTYiIGZpbGw9IiMwZDBkMTEiLz48Y2lyY2xlIGN4PSIzMiIgY3k9IjMyIiByPSIyMiIgZmlsbD0iIzFhMWEyMiIgc3Ryb2tlPSIjMmEyYTM2IiBzdHJva2Utd2lkdGg9IjIiLz48Y2lyY2xlIGN4PSIzMiIgY3k9IjIyIiByPSIxMSIgZmlsbD0iI2ZmMmQ1NSIgb3BhY2l0eT0iMC45Ii8+PGNpcmNsZSBjeD0iMjMiIGN5PSIzNyIgcj0iMTEiIGZpbGw9IiMwMGU2NzYiIG9wYWNpdHk9IjAuOSIvPjxjaXJjbGUgY3g9IjQxIiBjeT0iMzciIHI9IjExIiBmaWxsPSIjMDBlNWZmIiBvcGFjaXR5PSIwLjkiLz48Y2lyY2xlIGN4PSIzMiIgY3k9IjMyIiByPSI0IiBmaWxsPSIjZmZmZmZmIi8+PC9zdmc+">
    <title>AmbiHDR Proxy</title>
    <style>
        :root {
            --bg: #111111;
            --panel: #1d1d1d;
            --muted: #aaaaaa;
            --line: #2f2f2f;
            --text: #f4f4f4;
            --accent: #1ea7ff;
            --good: #4caf50;
            --bad: #f44336;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0; padding: 18px 12px; font-family: Arial, sans-serif;
            background: var(--bg); color: var(--text);
        }
        .card {
            max-width: 520px; margin: 0 auto; background: var(--panel); border-radius: 18px;
            padding: 20px 20px 24px; border: 1px solid var(--line);
        }
        h2 { margin: 0; text-align: center; }
        .subtitle { text-align: center; color: #7d7d7d; font-size: 11px; font-weight: bold; margin-bottom: 12px; }
        .status { text-align: center; margin: 12px 0 18px; font-size: 15px; color: var(--muted); }
        .toggle-btn {
            width: 100%; padding: 14px 12px; margin-bottom: 16px; border: none; font-size: 16px;
            border-radius: 10px; font-weight: bold; cursor: pointer; color: var(--text);
        }
        .toggle-btn.on { background: var(--good); }
        .toggle-btn.off { background: var(--bad); }

        .mode-toggle-container {
            display: flex; justify-content: space-between; align-items: center;
            background: #181818; padding: 10px 14px; border-radius: 12px;
            border: 1px solid var(--line); margin-bottom: 18px;
        }
        .mode-title { font-size: 14px; font-weight: bold; color: var(--text); }
        .segmented-control input { display: none; }
        .toggle-switch {
            position: relative; display: flex; align-items: center;
            width: 110px; height: 32px; background: #2a2a2a; border-radius: 16px;
            border: 1px solid var(--line); cursor: pointer; user-select: none;
        }
        .toggle-switch .opt-sdr, .toggle-switch .opt-hdr {
            flex: 1; text-align: center; font-size: 12px; font-weight: bold; z-index: 2;
            transition: color 0.2s; color: var(--muted);
        }
        .switch-handle {
            position: absolute; top: 2px; left: 2px; width: 52px; height: 26px;
            background: var(--accent); border-radius: 13px; transition: transform 0.25s ease; z-index: 1;
        }
        #hdr-mode-toggle:checked + .toggle-switch .switch-handle { transform: translateX(52px); }
        #hdr-mode-toggle:not(:checked) + .toggle-switch .opt-sdr { color: #fff; }
        #hdr-mode-toggle:checked + .toggle-switch .opt-hdr { color: #fff; }

        .hdr-h { color: #E6274E; }
        .hdr-d { color: #00E273; }
        .hdr-r { color: #02CFE6; }
        .tabs-nav { display: flex; border-bottom: 2px solid var(--line); margin-bottom: 18px; }
        .tab-btn {
            flex: 1; background: none; border: none; border-bottom: 2px solid transparent;
            margin-bottom: -2px; color: var(--muted); padding: 10px 0; font-size: 13px;
            font-weight: bold; cursor: pointer; transition: color 0.2s, border-color 0.2s;
        }
        .tab-btn:hover { color: var(--text); }
        .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); background: none; }

        .tab-panel { display: none; }
        .tab-panel.active { display: block; }
        
        .section-header {
            display: flex; justify-content: space-between; align-items: center;
            margin: 16px 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--line);
        }
        .section-title { font-size: 15px; font-weight: bold; }
        
        .mini-toggle {
            position: relative; display: inline-block; width: 44px; height: 22px;
        }
        .mini-toggle input { opacity: 0; width: 0; height: 0; }
        .mini-slider {
            position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
            background-color: #2a2a2a; border: 1px solid var(--line); transition: .2s; border-radius: 22px;
        }
        .mini-slider:before {
            position: absolute; content: ""; height: 16px; width: 16px; left: 2px; bottom: 2px;
            background-color: var(--muted); transition: .2s; border-radius: 50%;
        }
        input:checked + .mini-slider { background-color: var(--accent); }
        input:checked + .mini-slider:before { transform: translateX(22px); background-color: #fff; }

        .control-group { margin-bottom: 16px; }
        .control-group label {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 14px;
        }
        .control-group input[type=range], .control-group input[type=text], .control-group input[type=number], .control-group select {
            width: 100%;
        }
        .control-group input[type=text], .control-group input[type=number], .control-group select {
            background: #2a2a2a; border: 1px solid #444; color: var(--text); border-radius: 8px; padding: 9px 10px;
        }

        input[type=range] { accent-color: #e2e8f0; }
        input[type=range].slider-red { accent-color: #ff4d4d; }
        input[type=range].slider-green { accent-color: #4caf50; }
        input[type=range].slider-blue { accent-color: #2196f3; }

        .btn-apply, .btn-restore {
            width: 100%; padding: 10px 12px; border: none; border-radius: 8px; cursor: pointer; margin-top: 12px;
            font-weight: bold; color: var(--text);
        }
        .btn-apply { background: #0d79bb; }
        .btn-restore { background: #2a2a2a; border: 1px solid var(--line); color: #d9d9d9; }

        .perf-grid {
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 16px;
        }
        .perf-box {
            background: #181818; border: 1px solid var(--line); border-radius: 10px;
            padding: 10px 6px; text-align: center; transition: opacity 0.2s;
        }
        .perf-box.disabled { opacity: 0.4; }
        .perf-value { font-size: 18px; font-weight: bold; color: var(--accent); margin-top: 4px; }
        .perf-label { font-size: 11px; color: var(--muted); font-weight: bold; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Ambi<span class="hdr-h">H</span><span class="hdr-d">D</span><span class="hdr-r">R</span> Proxy</h2>
        <div class="subtitle">v0.2.4</div>

        <div class="status">
            Status: <b id="state-text" style="color: {{ '#4caf50' if config.PROXY_ACTIVE else '#f44336' }}">
                {{ 'Pass-through' if not config.PROXY_ACTIVE else ('HDR Correction' if config.ACTIVE_MODE == 'HDR' else 'SDR Correction') }}
            </b>
        </div>

        <button id="power-btn" class="toggle-btn {{ 'on' if config.PROXY_ACTIVE else 'off' }}" onclick="toggleProxy()">
            {{ 'Turn Off Proxy' if config.PROXY_ACTIVE else 'Turn On Proxy' }}
        </button>

        <div class="mode-toggle-container">
            <span class="mode-title">Active Mode</span>
            <div class="segmented-control">
                <input type="checkbox" id="hdr-mode-toggle" {{ 'checked' if config.ACTIVE_MODE == 'HDR' else '' }} onchange="toggleMode(this.checked)">
                <label for="hdr-mode-toggle" class="toggle-switch">
                    <span class="opt-sdr">SDR</span>
                    <span class="opt-hdr">HDR</span>
                    <span class="switch-handle"></span>
                </label>
            </div>
        </div>

        <div class="tabs-nav">
            <button class="tab-btn active" data-tab="sdr" onclick="showTab('sdr')">SDR Settings</button>
            <button class="tab-btn" data-tab="hdr" onclick="showTab('hdr')">HDR Settings</button>
            <button class="tab-btn" data-tab="settings" onclick="showTab('settings')">Setup</button>
        </div>

        <div id="tab-sdr" class="tab-panel active">
            <div class="section-header">
                <span class="section-title">SDR Correction</span>
            </div>
            <div class="control-group">
                <label>Exposure <span id="sdr-exp-val">{{ config.PROFILE_SDR.EXPOSURE }}</span></label>
                <input id="sdr-exp-slider" type="range" min="0.5" max="3.0" step="0.1" value="{{ config.PROFILE_SDR.EXPOSURE }}" oninput="updateProfile('sdr')">
            </div>
            <div class="control-group">
                <label>Gamma <span id="sdr-gamma-val">{{ config.PROFILE_SDR.GAMMA }}</span></label>
                <input id="sdr-gamma-slider" type="range" min="1.0" max="3.0" step="0.1" value="{{ config.PROFILE_SDR.GAMMA }}" oninput="updateProfile('sdr')">
            </div>
            <div class="control-group">
                <label>Saturation <span id="sdr-sat-val">{{ config.PROFILE_SDR.SATURATION }}</span></label>
                <input id="sdr-sat-slider" type="range" min="0.5" max="2.0" step="0.1" value="{{ config.PROFILE_SDR.SATURATION }}" oninput="updateProfile('sdr')">
            </div>
            <div class="control-group">
                <label>Black Cutoff <span id="sdr-cutoff-val">{{ config.PROFILE_SDR.BLACK_CUTOFF }}</span></label>
                <input id="sdr-cutoff-slider" type="range" min="0" max="25" step="1" value="{{ config.PROFILE_SDR.BLACK_CUTOFF }}" oninput="updateProfile('sdr')">
            </div>
            <div class="control-group">
                <label>Smoothing <span id="sdr-smooth-val">{{ config.PROFILE_SDR.SMOOTHING }}</span></label>
                <input id="sdr-smooth-slider" type="range" min="0.0" max="0.8" step="0.05" value="{{ config.PROFILE_SDR.SMOOTHING }}" oninput="updateProfile('sdr')">
            </div>
            <div class="section-header">
                <span class="section-title">Color Balance (RGB)</span>
            </div>
            <div class="control-group">
                <label>Red <span id="sdr-r-val">{{ config.PROFILE_SDR.GAIN_R }}</span></label>
                <input id="sdr-r-slider" class="slider-red" type="range" min="0.5" max="2.0" step="0.05" value="{{ config.PROFILE_SDR.GAIN_R }}" oninput="updateProfile('sdr')">
            </div>
            <div class="control-group">
                <label>Green <span id="sdr-g-val">{{ config.PROFILE_SDR.GAIN_G }}</span></label>
                <input id="sdr-g-slider" class="slider-green" type="range" min="0.5" max="2.0" step="0.05" value="{{ config.PROFILE_SDR.GAIN_G }}" oninput="updateProfile('sdr')">
            </div>
            <div class="control-group">
                <label>Blue <span id="sdr-b-val">{{ config.PROFILE_SDR.GAIN_B }}</span></label>
                <input id="sdr-b-slider" class="slider-blue" type="range" min="0.5" max="2.0" step="0.05" value="{{ config.PROFILE_SDR.GAIN_B }}" oninput="updateProfile('sdr')">
            </div>
            <button class="btn-restore" onclick="restoreProfileDefaults('sdr')">Restore SDR Defaults</button>
        </div>

        <div id="tab-hdr" class="tab-panel">
            <div class="section-header">
                <span class="section-title">HDR Correction</span>
            </div>
            <div class="control-group">
                <label>Input Color Space</label>
                <select id="hdr-color-space" onchange="updateProfile('hdr')">
                    <option value="REC2020" {{ 'selected' if config.PROFILE_HDR.INPUT_COLOR_SPACE == 'REC2020' else '' }}>Rec.2020 (HDR10 / Dolby Vision)</option>
                    <option value="DCIP3" {{ 'selected' if config.PROFILE_HDR.INPUT_COLOR_SPACE == 'DCIP3' else '' }}>DCI-P3 (Apple Display / Cinema)</option>
                </select>
            </div>
            <div class="control-group">
                <label>Exposure <span id="hdr-exp-val">{{ config.PROFILE_HDR.EXPOSURE }}</span></label>
                <input id="hdr-exp-slider" type="range" min="0.5" max="3.0" step="0.1" value="{{ config.PROFILE_HDR.EXPOSURE }}" oninput="updateProfile('hdr')">
            </div>
            <div class="control-group">
                <label>Gamma <span id="hdr-gamma-val">{{ config.PROFILE_HDR.GAMMA }}</span></label>
                <input id="hdr-gamma-slider" type="range" min="1.0" max="3.0" step="0.1" value="{{ config.PROFILE_HDR.GAMMA }}" oninput="updateProfile('hdr')">
            </div>
            <div class="control-group">
                <label>Saturation <span id="hdr-sat-val">{{ config.PROFILE_HDR.SATURATION }}</span></label>
                <input id="hdr-sat-slider" type="range" min="0.5" max="2.0" step="0.1" value="{{ config.PROFILE_HDR.SATURATION }}" oninput="updateProfile('hdr')">
            </div>
            <div class="control-group">
                <label>Black Cutoff <span id="hdr-cutoff-val">{{ config.PROFILE_HDR.BLACK_CUTOFF }}</span></label>
                <input id="hdr-cutoff-slider" type="range" min="0" max="25" step="1" value="{{ config.PROFILE_HDR.BLACK_CUTOFF }}" oninput="updateProfile('hdr')">
            </div>
            <div class="control-group">
                <label>Smoothing <span id="hdr-smooth-val">{{ config.PROFILE_HDR.SMOOTHING }}</span></label>
                <input id="hdr-smooth-slider" type="range" min="0.0" max="0.8" step="0.05" value="{{ config.PROFILE_HDR.SMOOTHING }}" oninput="updateProfile('hdr')">
            </div>
            <div class="section-header">
                <span class="section-title">Color Balance (RGB)</span>
            </div>
            <div class="control-group">
                <label>Red <span id="hdr-r-val">{{ config.PROFILE_HDR.GAIN_R }}</span></label>
                <input id="hdr-r-slider" class="slider-red" type="range" min="0.5" max="2.0" step="0.05" value="{{ config.PROFILE_HDR.GAIN_R }}" oninput="updateProfile('hdr')">
            </div>
            <div class="control-group">
                <label>Green <span id="hdr-g-val">{{ config.PROFILE_HDR.GAIN_G }}</span></label>
                <input id="hdr-g-slider" class="slider-green" type="range" min="0.5" max="2.0" step="0.05" value="{{ config.PROFILE_HDR.GAIN_G }}" oninput="updateProfile('hdr')">
            </div>
            <div class="control-group">
                <label>Blue <span id="hdr-b-val">{{ config.PROFILE_HDR.GAIN_B }}</span></label>
                <input id="hdr-b-slider" class="slider-blue" type="range" min="0.5" max="2.0" step="0.05" value="{{ config.PROFILE_HDR.GAIN_B }}" oninput="updateProfile('hdr')">
            </div>
            <button class="btn-restore" onclick="restoreProfileDefaults('hdr')">Restore HDR Defaults</button>
        </div>

        <div id="tab-settings" class="tab-panel">
            <div class="section-header">
                <span class="section-title">Performance Metrics</span>
                <label class="mini-toggle">
                    <input type="checkbox" id="perf-toggle" {{ 'checked' if config.PERF_MONITOR_ACTIVE else '' }} onchange="togglePerfMonitoring(this.checked)">
                    <span class="mini-slider"></span>
                </label>
            </div>
            <div class="perf-grid">
                <div class="perf-box" id="box-recv">
                    <div class="perf-label">RECV FPS</div>
                    <div id="perf-recv-fps" class="perf-value">0</div>
                </div>
                <div class="perf-box" id="box-proc">
                    <div class="perf-label">PROC FPS</div>
                    <div id="perf-proc-fps" class="perf-value">0</div>
                </div>
                <div class="perf-box" id="box-lat">
                    <div class="perf-label">LATENCY</div>
                    <div id="perf-latency" class="perf-value">0 ms</div>
                </div>
            </div>

            <div class="section-header">
                <span class="section-title">Network & Hardware Settings</span>
            </div>
            <div class="control-group">
                <label>WLED Target IP</label>
                <input id="settings-wled-ip" type="text" value="{{ config.WLED_IP }}">
            </div>
            <div class="control-group">
                <label>WLED Target Port</label>
                <input id="settings-wled-port" type="number" min="1" max="65535" value="{{ config.WLED_PORT }}">
            </div>
            <div class="control-group">
                <label>Number of LEDs</label>
                <input id="settings-leds" type="number" min="1" max="1000" value="{{ config.NUM_LEDS }}">
            </div>

            <button class="btn-apply" onclick="applySettings()">Save Settings</button>
        </div>
    </div>

    <script>
        const CONFIG = {{ config | tojson }};
        let activeProxyState = CONFIG.PROXY_ACTIVE;
        let activeModeState = CONFIG.ACTIVE_MODE;
        let perfMonitorActive = CONFIG.PERF_MONITOR_ACTIVE;
        let statsInterval = null;
        let debounceTimer;

        function showTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tab));
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === 'tab-' + tab));
        }

        function updateStatusUI() {
            const stateText = document.getElementById('state-text');
            if (!activeProxyState) {
                stateText.textContent = 'Pass-through';
                stateText.style.color = '#f44336';
            } else {
                stateText.textContent = activeModeState === 'HDR' ? 'HDR Correction' : 'SDR Correction';
                stateText.style.color = '#4caf50';
            }
        }

        function toggleMode(isHdr) {
            activeModeState = isHdr ? 'HDR' : 'SDR';
            updateStatusUI();
            
            fetch('/update_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ active_mode: activeModeState })
            });
        }

        function toggleProxy() {
            fetch('/toggle', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    activeProxyState = data.proxy_active;
                    const btn = document.getElementById('power-btn');
                    btn.classList.toggle('on', activeProxyState);
                    btn.classList.toggle('off', !activeProxyState);
                    btn.textContent = activeProxyState ? 'Turn Off Proxy' : 'Turn On Proxy';
                    updateStatusUI();
                });
        }

        function togglePerfMonitoring(enabled) {
            perfMonitorActive = enabled;
            fetch('/update_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ perf_monitor_active: enabled })
            });

            const boxes = [document.getElementById('box-recv'), document.getElementById('box-proc'), document.getElementById('box-lat')];

            if (enabled) {
                boxes.forEach(b => b.classList.remove('disabled'));
                fetchStats();
                if (!statsInterval) {
                    statsInterval = setInterval(fetchStats, 1000);
                }
            } else {
                boxes.forEach(b => b.classList.add('disabled'));
                if (statsInterval) {
                    clearInterval(statsInterval);
                    statsInterval = null;
                }
                document.getElementById('perf-recv-fps').textContent = 'OFF';
                document.getElementById('perf-proc-fps').textContent = 'OFF';
                document.getElementById('perf-latency').textContent = 'OFF';
            }
        }

        function updateProfileLabels(profileKey) {
            const suffixes = ['exp', 'gamma', 'sat', 'cutoff', 'smooth', 'r', 'g', 'b'];
            suffixes.forEach((suffix) => {
                const valueEl = document.getElementById(`${profileKey}-${suffix}-val`);
                if (valueEl) {
                    const slider = document.getElementById(`${profileKey}-${suffix}-slider`);
                    if (!slider) return;
                    const value = parseFloat(slider.value);
                    if (suffix === 'cutoff') valueEl.textContent = parseInt(value, 10);
                    else if (suffix === 'smooth') valueEl.textContent = value.toFixed(2);
                    else if (suffix === 'r' || suffix === 'g' || suffix === 'b') valueEl.textContent = value.toFixed(2);
                    else valueEl.textContent = value.toFixed(1);
                }
            });
        }

        function emitProfile(profileKey) {
            const payload = {};
            const profile = {
                exposure: parseFloat(document.getElementById(`${profileKey}-exp-slider`).value),
                gamma: parseFloat(document.getElementById(`${profileKey}-gamma-slider`).value),
                saturation: parseFloat(document.getElementById(`${profileKey}-sat-slider`).value),
                black_cutoff: parseInt(document.getElementById(`${profileKey}-cutoff-slider`).value, 10),
                smoothing: parseFloat(document.getElementById(`${profileKey}-smooth-slider`).value),
                gain_r: parseFloat(document.getElementById(`${profileKey}-r-slider`).value),
                gain_g: parseFloat(document.getElementById(`${profileKey}-g-slider`).value),
                gain_b: parseFloat(document.getElementById(`${profileKey}-b-slider`).value)
            };
            if (profileKey === 'hdr') {
                profile.input_color_space = document.getElementById('hdr-color-space').value;
            }
            payload[profileKey === 'sdr' ? 'profile_sdr' : 'profile_hdr'] = profile;
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                fetch('/update_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            }, 120);
            updateProfileLabels(profileKey);
        }

        function updateProfile(profileKey) {
            updateProfileLabels(profileKey);
            emitProfile(profileKey);
        }

        function applySettings() {
            const payload = {
                wled_ip: document.getElementById('settings-wled-ip').value.trim(),
                wled_port: parseInt(document.getElementById('settings-wled-port').value, 10) || 4048,
                num_leds: parseInt(document.getElementById('settings-leds').value, 10) || 227
            };
            fetch('/update_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).then(() => alert('Settings saved.'));
        }

        function restoreProfileDefaults(profileKey) {
            if (!confirm(`Restore default ${profileKey.toUpperCase()} settings?`)) return;
            const defaults = profileKey === 'sdr' ? {
                exposure: 1.2, gamma: 2.0, saturation: 1.1, black_cutoff: 5, smoothing: 0.0, gain_r: 1.0, gain_g: 1.0, gain_b: 1.0
            } : {
                exposure: 1.2, gamma: 2.0, saturation: 1.1, black_cutoff: 5, smoothing: 0.0, gain_r: 1.0, gain_g: 1.0, gain_b: 1.0, input_color_space: 'REC2020'
            };

            document.getElementById(`${profileKey}-exp-slider`).value = defaults.exposure;
            document.getElementById(`${profileKey}-gamma-slider`).value = defaults.gamma;
            document.getElementById(`${profileKey}-sat-slider`).value = defaults.saturation;
            document.getElementById(`${profileKey}-cutoff-slider`).value = defaults.black_cutoff;
            document.getElementById(`${profileKey}-smooth-slider`).value = defaults.smoothing;
            document.getElementById(`${profileKey}-r-slider`).value = defaults.gain_r;
            document.getElementById(`${profileKey}-g-slider`).value = defaults.gain_g;
            document.getElementById(`${profileKey}-b-slider`).value = defaults.gain_b;

            if (profileKey === 'hdr') {
                document.getElementById('hdr-color-space').value = defaults.input_color_space;
            }

            updateProfile(profileKey);
        }

        function fetchStats() {
            if (!perfMonitorActive) return;
            fetch('/stats')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('perf-recv-fps').textContent = data.received_fps;
                    document.getElementById('perf-proc-fps').textContent = data.processed_fps;
                    document.getElementById('perf-latency').textContent = `${data.latency_ms} ms`;
                })
                .catch(() => {});
        }

        document.addEventListener('DOMContentLoaded', () => {
            ['sdr', 'hdr'].forEach(updateProfileLabels);
            if (perfMonitorActive) {
                fetchStats();
                statsInterval = setInterval(fetchStats, 1000);
            } else {
                togglePerfMonitoring(false);
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    with config_lock:
        cfg_copy = config.copy()
    return render_template_string(HTML_UI, config=cfg_copy)

@app.route('/favicon.ico')
def favicon():
    return ('', 204)

@app.route('/stats')
def stats():
    with stats_lock:
        return jsonify(stats_data)

@app.route("/update_config", methods=["POST"])
def update_config():
    data = request.get_json() or {}

    with config_lock:
        if "active_mode" in data:
            mode = str(data["active_mode"]).upper()
            if mode in {"SDR", "HDR"}:
                config["ACTIVE_MODE"] = mode

        if "perf_monitor_active" in data:
            config["PERF_MONITOR_ACTIVE"] = bool(data["perf_monitor_active"])

        for target_key, payload_key in [("PROFILE_SDR", "profile_sdr"), ("PROFILE_HDR", "profile_hdr")]:
            if payload_key in data and isinstance(data[payload_key], dict):
                p_data = data[payload_key]
                mapped = {}
                for k, v in p_data.items():
                    key_upper = k.upper()
                    if key_upper == "BLACK_CUTOFF":
                        mapped[key_upper] = int(v)
                    elif key_upper == "INPUT_COLOR_SPACE":
                        val_upper = str(v).upper()
                        if target_key == "PROFILE_HDR" and val_upper not in {"REC2020", "DCIP3"}:
                            val_upper = "REC2020"
                        mapped[key_upper] = val_upper
                    else:
                        try:
                            mapped[key_upper] = float(v)
                        except (ValueError, TypeError):
                            mapped[key_upper] = v
                config[target_key].update(mapped)

        if "wled_ip" in data and data["wled_ip"]: config["WLED_IP"] = str(data["wled_ip"])
        if "wled_port" in data: config["WLED_PORT"] = int(data["wled_port"])
        if "num_leds" in data: config["NUM_LEDS"] = int(data["num_leds"])

        save_config(config)
        rebuild_lut_for_current_profile()

    return jsonify({"status": "ok", "active_mode": config.get("ACTIVE_MODE", "HDR")})

@app.route("/toggle", methods=["POST"])
def toggle():
    with config_lock:
        config["PROXY_ACTIVE"] = not config["PROXY_ACTIVE"]
        save_config(config)
        state = config["PROXY_ACTIVE"]
    return jsonify({"proxy_active": state, "active_mode": config.get("ACTIVE_MODE", "HDR")})

if __name__ == "__main__":
    log("Starting UDP loop on thread...")
    t = threading.Thread(target=udp_loop, daemon=True)
    t.start()
    log("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000)