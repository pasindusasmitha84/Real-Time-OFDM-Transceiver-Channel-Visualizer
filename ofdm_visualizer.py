#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Real-Time OFDM Transceiver & Channel Visualizer
#  Single-file PyQt5 + PyQtGraph application
# -----------------------------------------------------------------------------
#  REQUIRED INSTALLS (Python 3.8+):
#      pip install numpy
#      pip install scipy
#      pip install PyQt5
#      pip install pyqtgraph
#
#  or in one line:
#      pip install numpy scipy PyQt5 pyqtgraph
#
#  RUN:
#      python ofdm_visualizer.py              -> launches the GUI dashboard
#      python ofdm_visualizer.py --selftest   -> headless DSP verification run
# -----------------------------------------------------------------------------
#  SYSTEM SUMMARY
#   * 64 subcarriers, 16-sample cyclic prefix, Gray-coded QPSK on every
#     subcarrier.
#   * Frame = 1 block-pilot (preamble) OFDM symbol + 14 data OFDM symbols.
#   * Channel: 3-tap Rayleigh multipath (delays 0/3/7 samples, power-delay
#     profile 0/-3/-6 dB, unit total power) evolving frame-to-frame as a
#     Gauss-Markov process with Jakes correlation rho = J0(2*pi*fd*T_frame),
#     Carrier Frequency Offset (normalized to the subcarrier spacing), AWGN.
#   * Receiver: optional CP-correlation CFO estimation/correction, CP removal,
#     FFT, Least-Squares (LS) channel estimation on the preamble, one-tap
#     zero-forcing equalization, QPSK hard-decision demapping, BER and EVM.
#   * The DSP loop runs in a dedicated QThread paced at 60 FPS. The GUI thread
#     only renders. A frame back-pressure handshake guarantees the Qt event
#     queue never builds up, so the UI never freezes.
# =============================================================================

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
from scipy import signal, special

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


# =============================================================================
#  Global constants
# =============================================================================
APP_TITLE = "Real-Time OFDM Transceiver & Channel Visualizer"
TARGET_FPS = 60.0
QPSK_CONSTELLATION = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=np.complex128) / np.sqrt(2.0)


# =============================================================================
#  Configuration / parameter / result containers
# =============================================================================
@dataclass(frozen=True)
class OFDMConfig:
    """Static system configuration of the OFDM link."""
    n_subcarriers: int = 64
    cp_length: int = 16
    n_data_symbols: int = 14
    tap_delays: Tuple[int, ...] = (0, 3, 7)
    tap_powers_db: Tuple[float, ...] = (0.0, -3.0, -6.0)
    psd_nperseg: int = 128
    psd_nfft: int = 256
    psd_smoothing: float = 0.25
    seed: Optional[int] = None


@dataclass
class SimulationParameters:
    """Run-time parameters controlled live from the GUI."""
    snr_db: float = 20.0
    cfo: float = 0.0                 # normalized CFO, in units of subcarrier spacing
    multipath_enabled: bool = True
    equalizer_enabled: bool = True
    cfo_correction_enabled: bool = False
    doppler: float = 0.01            # normalized Doppler fd * T_frame


@dataclass
class FrameResult:
    """Everything the GUI needs to render one processed frame."""
    frame_index: int
    constellation: np.ndarray
    psd_freq: np.ndarray
    psd_tx_db: np.ndarray
    psd_rx_db: np.ndarray
    subcarrier_index: np.ndarray
    h_est: np.ndarray
    h_true: np.ndarray
    bit_errors: int
    total_bits: int
    ber: float
    evm_percent: float
    evm_db: float
    cfo_true: float
    cfo_est: float
    snr_db: float
    noise_variance: float
    equalizer_enabled: bool
    multipath_enabled: bool
    processing_time_ms: float
    engine_fps: float = 0.0
    dropped_frames: int = 0


# =============================================================================
#  Modulation helpers
# =============================================================================
def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Gray-coded QPSK: bit pair (b0, b1) -> ((1-2*b0) + j(1-2*b1)) / sqrt(2)."""
    bits = np.asarray(bits)
    if bits.shape[-1] != 2:
        raise ValueError("QPSK modulation requires the last bit dimension to be 2")
    in_phase = 1.0 - 2.0 * bits[..., 0].astype(np.float64)
    quadrature = 1.0 - 2.0 * bits[..., 1].astype(np.float64)
    return (in_phase + 1j * quadrature) / np.sqrt(2.0)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision Gray QPSK demapper (inverse of qpsk_modulate)."""
    symbols = np.asarray(symbols)
    bits = np.empty(symbols.shape + (2,), dtype=np.int8)
    bits[..., 0] = symbols.real < 0.0
    bits[..., 1] = symbols.imag < 0.0
    return bits


def theoretical_qpsk_ber(snr_db: float, rayleigh: bool) -> float:
    """Theoretical QPSK BER with perfect CSI. SNR is Es/N0 per subcarrier."""
    es_n0 = 10.0 ** (snr_db / 10.0)
    eb_n0 = es_n0 / 2.0
    if rayleigh:
        return float(0.5 * (1.0 - np.sqrt(eb_n0 / (1.0 + eb_n0))))
    return float(0.5 * special.erfc(np.sqrt(eb_n0)))


# =============================================================================
#  Channel model
# =============================================================================
class RayleighMultipathChannel:
    """Tapped-delay-line Rayleigh channel with Gauss-Markov (Jakes) evolution."""

    def __init__(self, delays, powers_db, rng: np.random.Generator):
        delays = np.asarray(delays, dtype=np.int64)
        powers_db = np.asarray(powers_db, dtype=np.float64)
        if delays.ndim != 1 or delays.size == 0:
            raise ValueError("Tap delays must be a non-empty 1-D sequence")
        if delays.shape != powers_db.shape:
            raise ValueError("Tap delays and tap powers must have equal length")
        if np.any(delays < 0):
            raise ValueError("Tap delays must be non-negative")
        linear_powers = 10.0 ** (powers_db / 10.0)
        linear_powers = linear_powers / np.sum(linear_powers)
        self.delays = delays
        self.tap_std = np.sqrt(linear_powers)
        self.max_delay = int(np.max(delays))
        self._rng = rng
        self._gains = self._draw_unit_gains()

    def _draw_unit_gains(self) -> np.ndarray:
        n_taps = self.delays.size
        return (self._rng.standard_normal(n_taps) + 1j * self._rng.standard_normal(n_taps)) / np.sqrt(2.0)

    def reset(self) -> None:
        """Draw a fresh, independent channel realization."""
        self._gains = self._draw_unit_gains()

    def evolve(self, doppler: float) -> None:
        """Advance one frame: g[k+1] = rho*g[k] + sqrt(1-rho^2)*w[k]."""
        rho = float(np.clip(special.j0(2.0 * np.pi * float(doppler)), -1.0, 1.0))
        innovation_gain = np.sqrt(max(0.0, 1.0 - rho * rho))
        self._gains = rho * self._gains + innovation_gain * self._draw_unit_gains()

    def impulse_response(self) -> np.ndarray:
        """Sample-spaced complex impulse response, length max_delay + 1."""
        h = np.zeros(self.max_delay + 1, dtype=np.complex128)
        np.add.at(h, self.delays, self.tap_std * self._gains)
        return h


# =============================================================================
#  OFDM transceiver engine
# =============================================================================
class OFDMEngine:
    """Complete OFDM TX -> channel -> RX chain operating on frames."""

    def __init__(self, config: Optional[OFDMConfig] = None):
        self.config = config if config is not None else OFDMConfig()
        cfg = self.config
        if cfg.n_subcarriers <= 0 or cfg.n_subcarriers % 2 != 0:
            raise ValueError("Number of subcarriers must be a positive even integer")
        if cfg.cp_length <= 0 or cfg.cp_length >= cfg.n_subcarriers:
            raise ValueError("Cyclic prefix must be positive and shorter than the symbol")
        if max(cfg.tap_delays) >= cfg.cp_length:
            raise ValueError("Maximum channel delay must be shorter than the cyclic prefix")

        self.n_fft = cfg.n_subcarriers
        self.cp = cfg.cp_length
        self.n_data = cfg.n_data_symbols
        self.n_symbols = 1 + self.n_data
        self.symbol_length = self.n_fft + self.cp
        self.frame_length = self.n_symbols * self.symbol_length
        self.bits_per_frame = self.n_data * self.n_fft * 2

        self._rng = np.random.default_rng(cfg.seed)
        pilot_rng = np.random.default_rng(0x0FD3)
        self.pilot_bits = pilot_rng.integers(0, 2, size=(self.n_fft, 2), dtype=np.int8)
        self.pilot_symbols = qpsk_modulate(self.pilot_bits)

        self.channel = RayleighMultipathChannel(cfg.tap_delays, cfg.tap_powers_db, self._rng)
        self.subcarrier_index = np.arange(-self.n_fft // 2, self.n_fft // 2)
        self._time_index = np.arange(self.frame_length, dtype=np.float64)
        self._cfo_phase = 0.0
        self._psd_tx_avg: Optional[np.ndarray] = None
        self._psd_rx_avg: Optional[np.ndarray] = None
        self._frame_index = 0

    # ----------------------------------------------------------------- TX ---
    def modulate_frame(self):
        """Random bits -> QPSK -> [pilot | data] grid -> IFFT -> add CP -> serialize."""
        bits = self._rng.integers(0, 2, size=(self.n_data, self.n_fft, 2), dtype=np.int8)
        data_symbols = qpsk_modulate(bits)
        grid = np.vstack([self.pilot_symbols[np.newaxis, :], data_symbols])
        time_symbols = np.fft.ifft(grid, axis=1) * np.sqrt(self.n_fft)
        with_cp = np.concatenate([time_symbols[:, -self.cp:], time_symbols], axis=1)
        return bits, data_symbols, with_cp.reshape(-1)

    # ------------------------------------------------------------ Channel ---
    def apply_channel(self, tx: np.ndarray, params: SimulationParameters):
        """Multipath fading -> CFO rotation -> AWGN."""
        if params.multipath_enabled:
            self.channel.evolve(params.doppler)
            h = self.channel.impulse_response()
        else:
            h = np.array([1.0 + 0.0j], dtype=np.complex128)

        faded = signal.lfilter(h, np.array([1.0]), tx)

        n = self._time_index[: tx.size]
        cfo = float(params.cfo)
        phase = self._cfo_phase + 2.0 * np.pi * cfo * n / self.n_fft
        rotated = faded * np.exp(1j * phase)
        self._cfo_phase = float(np.mod(self._cfo_phase + 2.0 * np.pi * cfo * tx.size / self.n_fft, 2.0 * np.pi))

        tx_power = float(np.mean(np.abs(tx) ** 2))
        noise_variance = tx_power / (10.0 ** (float(params.snr_db) / 10.0))
        noise = np.sqrt(noise_variance / 2.0) * (
            self._rng.standard_normal(tx.size) + 1j * self._rng.standard_normal(tx.size)
        )
        return rotated + noise, h, noise_variance

    # ----------------------------------------------------------------- RX ---
    def estimate_cfo(self, rx: np.ndarray, max_delay: int) -> float:
        """CP-correlation (van de Beek style) CFO estimator, range |eps| < 0.5.

        The first `max_delay` CP samples of each symbol carry ISI from the previous
        symbol, so they are skipped.
        """
        blocks = rx.reshape(self.n_symbols, self.symbol_length)
        start = int(min(max(int(max_delay), 0), self.cp - 1))
        head = blocks[:, start:self.cp]
        tail = blocks[:, start + self.n_fft:self.cp + self.n_fft]
        correlation = np.sum(np.conj(head) * tail)
        if np.abs(correlation) < 1e-15:
            return 0.0
        return float(np.angle(correlation) / (2.0 * np.pi))

    def correct_cfo(self, rx: np.ndarray, cfo_estimate: float) -> np.ndarray:
        n = self._time_index[: rx.size]
        return rx * np.exp(-1j * 2.0 * np.pi * cfo_estimate * n / self.n_fft)

    def demodulate_frame(self, rx: np.ndarray) -> np.ndarray:
        """Serial -> parallel, CP removal, FFT. Returns (n_symbols, n_fft) grid."""
        blocks = rx.reshape(self.n_symbols, self.symbol_length)[:, self.cp:]
        return np.fft.fft(blocks, axis=1) / np.sqrt(self.n_fft)

    def ls_channel_estimate(self, received_pilot: np.ndarray) -> np.ndarray:
        """Least-Squares estimate H_LS[k] = Y_p[k] / X_p[k]."""
        return received_pilot / self.pilot_symbols

    @staticmethod
    def equalize(received_data: np.ndarray, h_est: np.ndarray) -> np.ndarray:
        """One-tap zero-forcing equalizer per subcarrier."""
        safe_h = np.where(np.abs(h_est) < 1e-9, 1e-9 + 0.0j, h_est)
        return received_data / safe_h[np.newaxis, :]

    def _welch_psd(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        nperseg = min(cfg.psd_nperseg, x.size)
        freqs, psd = signal.welch(
            x,
            fs=1.0,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            nfft=max(cfg.psd_nfft, nperseg),
            detrend=False,
            return_onesided=False,
            scaling="density",
        )
        return np.fft.fftshift(freqs), np.fft.fftshift(np.real(psd))

    def _smooth(self, average: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
        alpha = float(np.clip(self.config.psd_smoothing, 0.0, 1.0))
        if average is None or average.shape != new.shape or alpha >= 1.0:
            return new.copy()
        return (1.0 - alpha) * average + alpha * new

    def compute_psd(self, tx: np.ndarray, rx: np.ndarray):
        freqs, psd_tx = self._welch_psd(tx)
        _, psd_rx = self._welch_psd(rx)
        self._psd_tx_avg = self._smooth(self._psd_tx_avg, psd_tx)
        self._psd_rx_avg = self._smooth(self._psd_rx_avg, psd_rx)
        freq_subcarriers = freqs * self.n_fft
        return (
            freq_subcarriers,
            10.0 * np.log10(self._psd_tx_avg + 1e-20),
            10.0 * np.log10(self._psd_rx_avg + 1e-20),
        )

    def reset_channel(self) -> None:
        self.channel.reset()

    # ------------------------------------------------------- Full chain ---
    def process_frame(self, params: SimulationParameters) -> FrameResult:
        t_start = time.perf_counter()

        bits, data_symbols, tx = self.modulate_frame()
        rx, h, noise_variance = self.apply_channel(tx, params)

        max_delay = self.channel.max_delay if params.multipath_enabled else 0
        cfo_estimate = self.estimate_cfo(rx, max_delay)
        rx_processed = self.correct_cfo(rx, cfo_estimate) if params.cfo_correction_enabled else rx

        grid = self.demodulate_frame(rx_processed)
        h_ls = self.ls_channel_estimate(grid[0])
        received_data = grid[1:]
        if params.equalizer_enabled:
            detected = self.equalize(received_data, h_ls)
        else:
            detected = received_data

        rx_bits = qpsk_demodulate(detected)
        bit_errors = int(np.count_nonzero(rx_bits != bits))
        total_bits = int(bits.size)
        ber = bit_errors / total_bits

        error_power = float(np.mean(np.abs(detected - data_symbols) ** 2))
        reference_power = float(np.mean(np.abs(data_symbols) ** 2))
        evm_rms = np.sqrt(error_power / reference_power) if reference_power > 0 else 0.0
        evm_db = 20.0 * np.log10(evm_rms) if evm_rms > 0 else -np.inf

        h_true = np.fft.fft(h, self.n_fft)
        psd_freq, psd_tx_db, psd_rx_db = self.compute_psd(tx, rx)

        self._frame_index += 1
        return FrameResult(
            frame_index=self._frame_index,
            constellation=detected.reshape(-1).copy(),
            psd_freq=psd_freq,
            psd_tx_db=psd_tx_db,
            psd_rx_db=psd_rx_db,
            subcarrier_index=self.subcarrier_index,
            h_est=np.fft.fftshift(h_ls),
            h_true=np.fft.fftshift(h_true),
            bit_errors=bit_errors,
            total_bits=total_bits,
            ber=float(ber),
            evm_percent=float(evm_rms * 100.0),
            evm_db=float(evm_db),
            cfo_true=float(params.cfo),
            cfo_est=float(cfo_estimate),
            snr_db=float(params.snr_db),
            noise_variance=float(noise_variance),
            equalizer_enabled=bool(params.equalizer_enabled),
            multipath_enabled=bool(params.multipath_enabled),
            processing_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )


# =============================================================================
#  Headless self-test (no GUI required beyond the imports)
# =============================================================================
def run_self_test(n_frames: int = 200) -> bool:
    engine = OFDMEngine(OFDMConfig(seed=2026))
    scenarios = [
        ("AWGN 30 dB | EQ on",
         SimulationParameters(snr_db=30.0, cfo=0.0, multipath_enabled=False, equalizer_enabled=True),
         lambda r: r["ber"] < 1e-4),
        ("AWGN 10 dB | EQ on",
         SimulationParameters(snr_db=10.0, cfo=0.0, multipath_enabled=False, equalizer_enabled=True),
         lambda r: r["ber"] < 5e-2),
        ("Rayleigh 30 dB | EQ on",
         SimulationParameters(snr_db=30.0, cfo=0.0, multipath_enabled=True, equalizer_enabled=True, doppler=0.05),
         lambda r: r["ber"] < 1e-2),
        ("Rayleigh 30 dB | EQ off",
         SimulationParameters(snr_db=30.0, cfo=0.0, multipath_enabled=True, equalizer_enabled=False, doppler=0.05),
         lambda r: r["ber"] > 0.1),
        ("AWGN 30 dB | CFO +0.20 | no correction",
         SimulationParameters(snr_db=30.0, cfo=0.2, multipath_enabled=False, equalizer_enabled=True,
                              cfo_correction_enabled=False),
         lambda r: r["ber"] > 0.05),
        ("AWGN 30 dB | CFO +0.20 | CP correction",
         SimulationParameters(snr_db=30.0, cfo=0.2, multipath_enabled=False, equalizer_enabled=True,
                              cfo_correction_enabled=True),
         lambda r: r["ber"] < 1e-3 and abs(r["cfo_est"] - 0.2) < 0.01),
        ("Rayleigh 25 dB | CFO -0.15 | CP correction",
         SimulationParameters(snr_db=25.0, cfo=-0.15, multipath_enabled=True, equalizer_enabled=True,
                              cfo_correction_enabled=True, doppler=0.05),
         lambda r: r["ber"] < 3e-2 and abs(r["cfo_est"] + 0.15) < 0.03),
    ]

    print(f"{APP_TITLE} - DSP self-test ({n_frames} frames/scenario, "
          f"{engine.bits_per_frame} bits/frame)")
    print("-" * 104)
    print(f"{'Scenario':44s} {'BER':>10s} {'Theory*':>10s} {'EVM %':>8s} {'CFO est':>9s} {'ms/frame':>9s}  Result")
    print("-" * 104)
    all_passed = True
    for name, params, check in scenarios:
        engine.reset_channel()
        errors = 0
        bits = 0
        evm_values = []
        cfo_values = []
        times = []
        for _ in range(int(n_frames)):
            result = engine.process_frame(params)
            errors += result.bit_errors
            bits += result.total_bits
            evm_values.append(result.evm_percent)
            cfo_values.append(result.cfo_est)
            times.append(result.processing_time_ms)
        summary = {
            "ber": errors / bits,
            "evm": float(np.median(evm_values)),
            "cfo_est": float(np.mean(cfo_values)),
            "ms": float(np.mean(times)),
        }
        passed = bool(check(summary))
        all_passed = all_passed and passed
        theory = theoretical_qpsk_ber(params.snr_db, params.multipath_enabled)
        print(f"{name:44s} {summary['ber']:10.3e} {theory:10.3e} {summary['evm']:8.2f} "
              f"{summary['cfo_est']:+9.4f} {summary['ms']:9.3f}  {'PASS' if passed else 'FAIL'}")
    print("-" * 104)
    print("* Theory = QPSK with perfect CSI; LS estimation on one pilot symbol costs ~3 dB.")
    print("SELF-TEST " + ("PASSED" if all_passed else "FAILED"))
    return all_passed


# =============================================================================
#  Processing thread
# =============================================================================
class OFDMWorker(QtCore.QThread):
    """Runs the OFDM chain continuously at TARGET_FPS in its own thread."""

    frameReady = QtCore.pyqtSignal(object)
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(self, engine: OFDMEngine, target_fps: float = TARGET_FPS, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._params = SimulationParameters()
        self._params_lock = threading.Lock()
        self._running = threading.Event()
        self._paused = threading.Event()
        self._reset_channel_requested = threading.Event()
        self._gui_ready = threading.Event()
        self._gui_ready.set()
        self._target_period = 1.0 / max(1.0, float(target_fps))
        self._dropped_frames = 0

    # ---- thread-safe control API (called from the GUI thread) ----
    def set_parameters(self, params: SimulationParameters) -> None:
        with self._params_lock:
            self._params = replace(params)

    def request_channel_reset(self) -> None:
        self._reset_channel_requested.set()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def acknowledge_frame(self) -> None:
        """GUI calls this after rendering; allows the next frame to be emitted."""
        self._gui_ready.set()

    def stop(self) -> None:
        self._running.clear()
        self._paused.clear()
        if not self.wait(3000):
            self.terminate()
            self.wait()

    # ---- thread body ----
    def run(self) -> None:
        self._running.set()
        next_deadline = time.perf_counter()
        fps_window_start = next_deadline
        fps_frame_count = 0
        measured_fps = 0.0

        while self._running.is_set():
            if self._paused.is_set():
                time.sleep(0.02)
                next_deadline = time.perf_counter()
                fps_window_start = next_deadline
                fps_frame_count = 0
                continue

            with self._params_lock:
                params = replace(self._params)

            if self._reset_channel_requested.is_set():
                self._reset_channel_requested.clear()
                self._engine.reset_channel()

            try:
                result = self._engine.process_frame(params)
            except Exception as exc:  # report and keep the thread alive
                self.errorOccurred.emit(f"DSP error: {type(exc).__name__}: {exc}")
                time.sleep(0.25)
                next_deadline = time.perf_counter()
                continue

            fps_frame_count += 1
            now = time.perf_counter()
            elapsed = now - fps_window_start
            if elapsed >= 0.5:
                measured_fps = fps_frame_count / elapsed
                fps_window_start = now
                fps_frame_count = 0

            if self._gui_ready.is_set():
                self._gui_ready.clear()
                result.engine_fps = measured_fps
                result.dropped_frames = self._dropped_frames
                self.frameReady.emit(result)
            else:
                self._dropped_frames += 1

            next_deadline += self._target_period
            delay = next_deadline - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_deadline = time.perf_counter()


# =============================================================================
#  GUI
# =============================================================================
def apply_dark_palette(app: QtWidgets.QApplication) -> None:
    palette = QtGui.QPalette()
    window = QtGui.QColor(30, 30, 38)
    base = QtGui.QColor(22, 22, 28)
    text = QtGui.QColor(225, 225, 230)
    highlight = QtGui.QColor(0, 150, 220)
    palette.setColor(QtGui.QPalette.Window, window)
    palette.setColor(QtGui.QPalette.WindowText, text)
    palette.setColor(QtGui.QPalette.Base, base)
    palette.setColor(QtGui.QPalette.AlternateBase, window)
    palette.setColor(QtGui.QPalette.ToolTipBase, window)
    palette.setColor(QtGui.QPalette.ToolTipText, text)
    palette.setColor(QtGui.QPalette.Text, text)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 55))
    palette.setColor(QtGui.QPalette.ButtonText, text)
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 80, 80))
    palette.setColor(QtGui.QPalette.Highlight, highlight)
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(120, 120, 120))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(120, 120, 120))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor(120, 120, 120))
    app.setPalette(palette)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_TITLE)
        self.resize(1560, 940)

        self.engine = OFDMEngine(OFDMConfig())
        self.worker = OFDMWorker(self.engine, target_fps=TARGET_FPS)

        self._constellation_history = deque(maxlen=3)
        self._accumulated_errors = 0
        self._accumulated_bits = 0
        self._gui_frame_count = 0
        self._gui_fps_window_start = time.perf_counter()
        self._gui_fps = 0.0
        self.metric_labels = {}

        self._build_ui()
        self._connect_signals()
        self._on_controls_changed()
        self.worker.start()

    # ------------------------------------------------------------- layout ---
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)
        root_layout.addWidget(self._build_control_panel())
        root_layout.addWidget(self._build_plot_area(), stretch=1)
        self.statusBar().showMessage(
            f"OFDM: N={self.engine.n_fft}, CP={self.engine.cp}, QPSK, "
            f"1 pilot + {self.engine.n_data} data symbols/frame | target {TARGET_FPS:.0f} FPS"
        )

    def _build_control_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("OFDM Transceiver Control")
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 3)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        description = QtWidgets.QLabel(
            "Live 64-subcarrier QPSK OFDM link. Adjust the channel on the left and "
            "watch the constellation, spectrum and LS channel estimate react in real time."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # ---- channel group ----
        channel_group = QtWidgets.QGroupBox("Channel Model")
        channel_layout = QtWidgets.QVBoxLayout(channel_group)

        self.snr_label = QtWidgets.QLabel()
        self.snr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.snr_slider.setRange(0, 30)
        self.snr_slider.setValue(20)
        self.snr_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.snr_slider.setTickInterval(5)
        self.snr_slider.setToolTip("AWGN signal-to-noise ratio (Es/N0 per subcarrier)")
        channel_layout.addWidget(self.snr_label)
        channel_layout.addWidget(self.snr_slider)

        self.cfo_label = QtWidgets.QLabel()
        self.cfo_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cfo_slider.setRange(-50, 50)
        self.cfo_slider.setValue(0)
        self.cfo_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.cfo_slider.setTickInterval(10)
        self.cfo_slider.setToolTip("Carrier frequency offset, normalized to the subcarrier spacing")
        channel_layout.addWidget(self.cfo_label)
        channel_layout.addWidget(self.cfo_slider)

        self.multipath_checkbox = QtWidgets.QCheckBox("Multipath Fading (3-tap Rayleigh)")
        self.multipath_checkbox.setChecked(True)
        channel_layout.addWidget(self.multipath_checkbox)

        self.doppler_label = QtWidgets.QLabel()
        self.doppler_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.doppler_slider.setRange(0, 50)
        self.doppler_slider.setValue(10)
        self.doppler_slider.setToolTip("Fading speed: normalized Doppler fd x frame duration (0 = frozen)")
        channel_layout.addWidget(self.doppler_label)
        channel_layout.addWidget(self.doppler_slider)

        self.new_channel_button = QtWidgets.QPushButton("New Channel Realization")
        channel_layout.addWidget(self.new_channel_button)
        layout.addWidget(channel_group)

        # ---- receiver group ----
        receiver_group = QtWidgets.QGroupBox("Receiver")
        receiver_layout = QtWidgets.QVBoxLayout(receiver_group)
        self.equalizer_checkbox = QtWidgets.QCheckBox("LS Equalizer ON")
        self.equalizer_checkbox.setChecked(True)
        self.cfo_correction_checkbox = QtWidgets.QCheckBox("CP-based CFO Correction")
        self.cfo_correction_checkbox.setChecked(False)
        receiver_layout.addWidget(self.equalizer_checkbox)
        receiver_layout.addWidget(self.cfo_correction_checkbox)
        layout.addWidget(receiver_group)

        # ---- display / run group ----
        display_group = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QFormLayout(display_group)
        self.persistence_spin = QtWidgets.QSpinBox()
        self.persistence_spin.setRange(1, 10)
        self.persistence_spin.setValue(3)
        self.persistence_spin.setSuffix(" frames")
        display_layout.addRow("Constellation persistence:", self.persistence_spin)
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.setCheckable(True)
        display_layout.addRow(self.pause_button)
        layout.addWidget(display_group)

        # ---- metrics group ----
        metrics_group = QtWidgets.QGroupBox("Link Metrics")
        metrics_layout = QtWidgets.QFormLayout(metrics_group)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        for key, caption in [
            ("ber_frame", "BER (frame):"),
            ("ber_acc", "BER (accumulated):"),
            ("ber_theory", "BER theory (perfect CSI):"),
            ("evm", "EVM:"),
            ("cfo_true", "CFO applied:"),
            ("cfo_est", "CFO estimated (CP):"),
            ("frame", "Frame #:"),
            ("proc", "DSP time / frame:"),
            ("engine_fps", "DSP thread FPS:"),
            ("gui_fps", "GUI render FPS:"),
            ("dropped", "Frames skipped by GUI:"),
        ]:
            value_label = QtWidgets.QLabel("-")
            value_label.setFont(mono)
            value_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self.metric_labels[key] = value_label
            metrics_layout.addRow(caption, value_label)
        layout.addWidget(metrics_group)

        layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        return scroll

    @staticmethod
    def _configure_plot(plot, x_label: str, y_label: str) -> None:
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setMenuEnabled(True)

    def _build_plot_area(self) -> QtWidgets.QWidget:
        self.plot_widget = pg.GraphicsLayoutWidget()

        # ---- constellation ----
        self.constellation_plot = self.plot_widget.addPlot(row=0, col=0, title="Subcarrier Constellation")
        self._configure_plot(self.constellation_plot, "In-Phase (I)", "Quadrature (Q)")
        self.constellation_plot.setAspectLocked(True)
        self.constellation_plot.setXRange(-2.0, 2.0, padding=0.0)
        self.constellation_plot.setYRange(-2.0, 2.0, padding=0.0)
        self.constellation_points = self.constellation_plot.plot(
            np.zeros(0), np.zeros(0),
            pen=None, symbol="o", symbolSize=3,
            symbolBrush=pg.mkBrush(0, 200, 255, 150), symbolPen=None,
        )
        self.ideal_points = self.constellation_plot.plot(
            QPSK_CONSTELLATION.real, QPSK_CONSTELLATION.imag,
            pen=None, symbol="+", symbolSize=18,
            symbolBrush=pg.mkBrush(255, 70, 70), symbolPen=pg.mkPen(255, 70, 70, width=2),
        )

        # ---- PSD ----
        self.psd_plot = self.plot_widget.addPlot(row=0, col=1, title="Power Spectral Density (Welch FFT)")
        self._configure_plot(self.psd_plot, "Frequency (subcarrier spacings)", "PSD (dB, normalized)")
        self.psd_plot.addLegend(offset=(-10, 10))
        self.psd_tx_curve = self.psd_plot.plot(pen=pg.mkPen(150, 150, 150, width=1), name="TX")
        self.psd_rx_curve = self.psd_plot.plot(pen=pg.mkPen(0, 255, 130, width=2), name="RX (channel + CFO + AWGN)")
        self.psd_plot.setXRange(-self.engine.n_fft / 2, self.engine.n_fft / 2, padding=0.0)
        self.psd_plot.setYRange(-40.0, 15.0, padding=0.0)

        # ---- channel magnitude ----
        self.mag_plot = self.plot_widget.addPlot(row=1, col=0, title="Channel Frequency Response |H(k)|")
        self._configure_plot(self.mag_plot, "Subcarrier index k", "|H| (dB)")
        self.mag_plot.addLegend(offset=(-10, 10))
        self.h_true_mag_curve = self.mag_plot.plot(pen=pg.mkPen(255, 210, 0, width=2), name="True channel")
        self.h_est_mag_curve = self.mag_plot.plot(
            pen=pg.mkPen(0, 200, 255, width=1), symbol="o", symbolSize=5,
            symbolBrush=pg.mkBrush(0, 200, 255, 200), symbolPen=None, name="LS estimate",
        )
        self.mag_plot.setXRange(-self.engine.n_fft / 2, self.engine.n_fft / 2 - 1, padding=0.02)
        self.mag_plot.setYRange(-30.0, 12.0, padding=0.0)

        # ---- channel phase ----
        self.phase_plot = self.plot_widget.addPlot(row=1, col=1, title="Channel Phase Response \u2220H(k)")
        self._configure_plot(self.phase_plot, "Subcarrier index k", "Phase (degrees)")
        self.phase_plot.addLegend(offset=(-10, 10))
        self.h_true_phase_curve = self.phase_plot.plot(pen=pg.mkPen(255, 210, 0, width=2), name="True channel")
        self.h_est_phase_curve = self.phase_plot.plot(
            pen=None, symbol="o", symbolSize=5,
            symbolBrush=pg.mkBrush(0, 200, 255, 200), symbolPen=None, name="LS estimate",
        )
        self.phase_plot.setXRange(-self.engine.n_fft / 2, self.engine.n_fft / 2 - 1, padding=0.02)
        self.phase_plot.setYRange(-180.0, 180.0, padding=0.02)

        return self.plot_widget

    # ------------------------------------------------------------ signals ---
    def _connect_signals(self) -> None:
        self.snr_slider.valueChanged.connect(self._on_controls_changed)
        self.cfo_slider.valueChanged.connect(self._on_controls_changed)
        self.doppler_slider.valueChanged.connect(self._on_controls_changed)
        self.multipath_checkbox.toggled.connect(self._on_controls_changed)
        self.equalizer_checkbox.toggled.connect(self._on_controls_changed)
        self.cfo_correction_checkbox.toggled.connect(self._on_controls_changed)
        self.new_channel_button.clicked.connect(self._on_new_channel)
        self.persistence_spin.valueChanged.connect(self._on_persistence_changed)
        self.pause_button.toggled.connect(self._on_pause_toggled)
        self.worker.frameReady.connect(self._on_frame_ready)
        self.worker.errorOccurred.connect(self._on_worker_error)

    def _current_parameters(self) -> SimulationParameters:
        return SimulationParameters(
            snr_db=float(self.snr_slider.value()),
            cfo=self.cfo_slider.value() / 100.0,
            multipath_enabled=self.multipath_checkbox.isChecked(),
            equalizer_enabled=self.equalizer_checkbox.isChecked(),
            cfo_correction_enabled=self.cfo_correction_checkbox.isChecked(),
            doppler=self.doppler_slider.value() / 1000.0,
        )

    def _on_controls_changed(self, *_args) -> None:
        params = self._current_parameters()
        self.snr_label.setText(f"SNR: {params.snr_db:.0f} dB")
        self.cfo_label.setText(f"CFO \u03b5: {params.cfo:+.2f} \u00d7 subcarrier spacing")
        self.doppler_label.setText(f"Fading speed fd\u00b7T_frame: {params.doppler:.3f}")
        self.doppler_slider.setEnabled(params.multipath_enabled)
        self.doppler_label.setEnabled(params.multipath_enabled)
        self.new_channel_button.setEnabled(params.multipath_enabled)
        self.constellation_plot.setTitle(
            "Subcarrier Constellation (" + ("LS-equalized" if params.equalizer_enabled else "raw, unequalized") + ")"
        )
        self.metric_labels["ber_theory"].setText(
            f"{theoretical_qpsk_ber(params.snr_db, params.multipath_enabled):.3e} "
            f"({'Rayleigh' if params.multipath_enabled else 'AWGN'})"
        )
        self.worker.set_parameters(params)
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._accumulated_errors = 0
        self._accumulated_bits = 0
        self._constellation_history.clear()

    def _on_new_channel(self) -> None:
        self.worker.request_channel_reset()
        self._reset_accumulators()

    def _on_persistence_changed(self, value: int) -> None:
        self._constellation_history = deque(self._constellation_history, maxlen=int(value))

    def _on_pause_toggled(self, paused: bool) -> None:
        self.worker.set_paused(paused)
        self.pause_button.setText("Resume" if paused else "Pause")
        self.statusBar().showMessage("Simulation paused" if paused else "Simulation running", 3000)

    def _on_worker_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    # ---------------------------------------------------------- rendering ---
    def _on_frame_ready(self, result: FrameResult) -> None:
        try:
            self._render(result)
        finally:
            self.worker.acknowledge_frame()

    def _render(self, result: FrameResult) -> None:
        # Constellation with persistence
        self._constellation_history.append(result.constellation)
        points = np.concatenate(list(self._constellation_history))
        self.constellation_points.setData(points.real, points.imag)

        # PSD
        self.psd_tx_curve.setData(result.psd_freq, result.psd_tx_db)
        self.psd_rx_curve.setData(result.psd_freq, result.psd_rx_db)

        # Channel frequency response
        k = result.subcarrier_index
        self.h_true_mag_curve.setData(k, 20.0 * np.log10(np.abs(result.h_true) + 1e-12))
        self.h_est_mag_curve.setData(k, 20.0 * np.log10(np.abs(result.h_est) + 1e-12))
        self.h_true_phase_curve.setData(k, np.degrees(np.angle(result.h_true)))
        self.h_est_phase_curve.setData(k, np.degrees(np.angle(result.h_est)))

        # Metrics
        self._accumulated_errors += result.bit_errors
        self._accumulated_bits += result.total_bits
        accumulated_ber = self._accumulated_errors / self._accumulated_bits if self._accumulated_bits else 0.0
        labels = self.metric_labels
        labels["ber_frame"].setText(f"{result.ber:.3e}  ({result.bit_errors}/{result.total_bits})")
        labels["ber_acc"].setText(f"{accumulated_ber:.3e}  ({self._accumulated_bits:,} bits)")
        evm_db_text = f"{result.evm_db:.1f} dB" if np.isfinite(result.evm_db) else "-inf dB"
        labels["evm"].setText(f"{result.evm_percent:.2f} %  ({evm_db_text})")
        labels["cfo_true"].setText(f"{result.cfo_true:+.4f}")
        labels["cfo_est"].setText(f"{result.cfo_est:+.4f}")
        labels["frame"].setText(f"{result.frame_index}")
        labels["proc"].setText(f"{result.processing_time_ms:.2f} ms")
        labels["engine_fps"].setText(f"{result.engine_fps:.1f}")
        labels["dropped"].setText(f"{result.dropped_frames}")

        self._gui_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._gui_fps_window_start
        if elapsed >= 0.5:
            self._gui_fps = self._gui_frame_count / elapsed
            self._gui_frame_count = 0
            self._gui_fps_window_start = now
        labels["gui_fps"].setText(f"{self._gui_fps:.1f}")

    # ------------------------------------------------------------ teardown ---
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.worker.stop()
        event.accept()


# =============================================================================
#  Entry point
# =============================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--selftest", action="store_true", help="run the headless DSP self-test and exit")
    parser.add_argument("--frames", type=int, default=200, help="frames per self-test scenario")
    args, qt_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    if args.selftest:
        return 0 if run_self_test(args.frames) else 1

    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication([sys.argv[0]] + qt_args)
    app.setApplicationName(APP_TITLE)
    app.setStyle("Fusion")
    apply_dark_palette(app)
    pg.setConfigOptions(antialias=False, background=(18, 18, 24), foreground=(220, 220, 225))

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
