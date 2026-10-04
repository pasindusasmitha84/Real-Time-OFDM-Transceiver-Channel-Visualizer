# Real-Time OFDM Transceiver & Channel Visualizer

A high-performance, multithreaded 64-subcarrier QPSK OFDM transceiver and Rayleigh multipath channel visualizer built in Python. Designed with a decoupled architecture where a dedicated `QThread` runs the digital signal processing (DSP) pipeline at a stable 60 FPS while the GUI thread handles real-time rendering without blocking.

---

### 🚀 Key Features

* **Complete Physical Layer Chain:** Random bit generation, Gray-coded QPSK modulation, pilot insertion, IFFT, cyclic prefix (CP) insertion, tapped-delay-line Rayleigh fading channel, Carrier Frequency Offset (CFO), and AWGN.
* **Advanced Receiver & Estimation:** CP-correlation CFO estimation (van de Beek algorithm), CP removal, FFT, Least-Squares (LS) channel estimation on preamble symbols, and one-tap zero-forcing equalization.
* **Real-Time Interactive Dashboard (PyQt5 + PyQtGraph):**
  * Live subcarrier constellation scatter plot with configurable persistence.
  * Real-time Power Spectral Density (PSD) estimation via Welch's method.
  * Instantaneous tracking of true vs. estimated channel frequency response magnitude and phase.
* **Live Link Metrics:** Frame-by-frame and accumulated Bit Error Rate (BER), Error Vector Magnitude (EVM in % and dB), DSP processing time per frame, and UI frame-rate counters.
* **Headless Verification Mode:** Includes an automated DSP self-test suite (`--selftest`) validating link performance against theoretical QPSK curves across various SNR, fading, and CFO scenarios.

---

### 🛠️ Installation & Requirements

Requires **Python 3.8+**. Install dependencies via pip:

```bash
pip install numpy scipy PyQt5 pyqtgraph
