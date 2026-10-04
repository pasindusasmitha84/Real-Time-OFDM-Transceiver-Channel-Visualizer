# Real-Time-OFDM-Transceiver-Channel-Visualizer
A high-performance, multithreaded 64-subcarrier QPSK OFDM transceiver and Rayleigh multipath channel visualizer built in Python. Designed with a decoupled architecture where a dedicated `QThread` runs the digital signal processing (DSP) pipeline at a stable 60 FPS while the GUI thread handles real-time rendering without blocking.
