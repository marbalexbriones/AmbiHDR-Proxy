# AmbiHDR Proxy

**AmbiHDR Proxy** is a high-performance UDP proxy written in Python designed to intercept, transform, and real-time correct ambient lighting (**Ambilight**) data streams sent via the **DDP** (Distributed Display Protocol) to **WLED** controllers.

It resolves desaturation, washed-out colors, and overexposure issues that occur when displaying **HDR** content (Rec.2020 / DCI-P3) on RGB LED strips operating in the sRGB/Rec.709 color space.

---

## Key Features

* **Real-Time Tone Mapping & Color Space Conversion**:
  * Matrix transformations from **Rec.2020** (HDR10 / Dolby Vision) and **DCI-P3** to **Rec.709** (sRGB).
  * Dynamic 3D Look-Up Table (**3D LUT** of 64 x 64 x 64) generation running in a background worker thread to prevent main thread blocking.
* **Ultra-Low Latency (~1 ms)**:
  * Vectorized matrix processing using `NumPy`, executing color corrections via byte-level bit-shifting.
* **Non-Blocking UDP Socket Management**:
  * Intelligent buffer handling with frame dropping to prevent latency accumulation or desynchronization caused by network jitter.
* **Interactive Web Control Panel (Flask)**:
  * On-the-fly configuration of independent profiles for **SDR** and **HDR**.
  * Fine-tuning for Exposure, Gamma, Saturation, Black Cutoff, Temporal Smoothing, and Per-Channel RGB Gain.
* **Built-in Performance Metrics**:
  * Live monitoring of Reception FPS (RECV FPS), Processing FPS (PROC FPS), and internal processing latency.

---

## Data Flow Architecture

```text
[ Capture Source ] ---> UDP / DDP (Port 21324) ---> [ AmbiHDR Proxy ] ---> UDP / DDP (Port 4048) ---> [ ESP32 / WLED ]
(ScreenGlow / PC)                                   (LUT Processing)                                  (RGB LED Strip)
