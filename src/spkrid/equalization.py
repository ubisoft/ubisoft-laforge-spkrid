"""
This module provides functions for audio signal processing, including pre-emphasis and de-emphasis filtering,
"""

import numpy as np
import scipy


def de_emphasis(samples, pre_emphasis_coeff=0.97):
    """
    Apply a de-emphasis filter to the audio signal.

    Args:
        samples (numpy.ndarray): The pre-emphasized audio signal samples.
        pre_emphasis_coeff (float): The pre-emphasis coefficient. Default is 0.97.

    Returns:
        numpy.ndarray: The de-emphasized audio signal.
    """
    de_emphasized_samples = scipy.signal.lfilter([1], [1, -pre_emphasis_coeff], samples)
    return de_emphasized_samples


def pre_emphasis(samples, pre_emphasis_coeff=0.97):
    """
    Apply a pre-emphasis filter to the audio signal.

    Args:
        samples (numpy.ndarray): The audio signal samples.
        pre_emphasis_coeff (float): The pre-emphasis
        coefficient. Default is 0.97.

    """
    pre_emphasized_samples = scipy.signal.lfilter(
        [1, -pre_emphasis_coeff], [1], samples
    )
    return pre_emphasized_samples


def apply_filter(signal, freqs, gains, n_filt_coefs=127, n_bands=None):
    """
    Apply a FIR filter to a signal using a defined frequency response
    Args:
        freqs (array-like): The frequencies at which the gain is defined.
        gains (array-like): The gain at each frequency.
        n_filt_coefs (int): The number of filter coefficients to use. Must be odd.
                        A bigger number will allow for more smoothing but will be slower to compute.
        n_bands (int): The number of bands to use for the filter. Those bands will be logarithmically spaced in frequency.
                   If None, the raw (not averaged) input frequency response will be used
    Returns:
        filtered_sig (array-like): The filtered signal.
        he (array-like): The time-domain FIR filter coefficients.
    """

    if n_bands is not None:
        # Compute the nb of points in each band (log spaced)
        n_pts_orig = n_filt_coefs
        n_pts_new = n_bands
        p_start = 1
        p_end = n_pts_orig / 4
        r = (p_end / p_start) ** (1.0 / n_pts_new)

        interv = p_start * r ** np.arange(n_pts_new)

        interv = n_pts_orig * interv / np.sum(interv)
        interv = np.clip(np.round(interv).astype(int), 1, None)

        # Compute the mean frequency and gain in each band
        freqlogs = [freqs[0]]
        gainslogs = [gains[0]]
        freq_idx = 0
        k = 0
        while freq_idx < len(freqs) - 1:
            freq_idx_next = freq_idx + interv[k]
            mean_freq = np.mean(freqs[freq_idx:freq_idx_next])
            mean_gain = np.mean(gains[freq_idx:freq_idx_next])
            freqlogs.append(mean_freq)
            gainslogs.append(mean_gain)
            freq_idx = freq_idx_next
            k += 1

        freqlogs.append(freqs[-1])
        gainslogs.append(gains[-1])

        if freqlogs[1] == 0:
            freqlogs = freqlogs[1:]
            gainslogs = gainslogs[1:]

        freqs = np.array(freqlogs)
        gains = np.array(gainslogs)

    # Compute the time-domain filter coefficients
    he = scipy.signal.firwin2(
        n_filt_coefs, freqs, np.sqrt(gains), antisymmetric=False, fs=1.0
    )

    # Apply the filter to the noisy signal
    filtered_sig = scipy.signal.filtfilt(he, [1.0], signal)

    return filtered_sig, he


def filter_welch(sig_ref, sig_in, n_filt_coefs=511, filter_power=1.0, n_bands=16):
    """
    Filters sig_in so it has a spectrum that resembles sig_ref, using Welch's method to compute the PSD
    Args:
        sig_ref (array-like): A reference signal
        sig_in (array-like): The input signal to be filtered.
        n_filt_coefs (int): The number of filter coefficients to use. Must be odd.
                            A bigger number will allow for more smoothing but will be slower to compute.
        filter_power (float): The global stregth of the filter. A value of 1.0 will give the standard Wiener filter.
                            A value of 2.0 will have the effect of applying the filter twice.
        n_bands (int): The number of bands to use for the filter. If not None, the filter will be divided into n_filt_coefs bands.
    Returns:
        wf (array-like): The filtered signal.
        he (array-like): The time-domain FIR filter coefficients.

    Note:
        The filter is computed in the frequency domain using the power spectral densities instead of the autocorrelation functions.
        This proved to be more stable
    """
    nperseg = (
        n_filt_coefs + 1
    )  # Number of samples per segment for the spectral density estimation

    # Compute the power spectral density of the clean and noisy signals
    f, Sw = scipy.signal.welch(
        sig_in,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        return_onesided=True,
        scaling="spectrum",
        average="mean",
        detrend=False,
    )
    f, Ss = scipy.signal.welch(
        sig_ref,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        return_onesided=True,
        scaling="spectrum",
        average="mean",
        detrend=False,
    )

    # Compute the Wiener filter transfer function in frequency domain
    H = (Ss / Sw) ** (filter_power / 2.0)

    return apply_filter(sig_in, f, H, n_filt_coefs, n_bands)
