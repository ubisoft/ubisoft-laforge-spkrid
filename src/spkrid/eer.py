"""
This module contains functions to compute the Equal Error Rate (EER).
"""

import itertools
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

from spkrid.speaker_embedding import get_speaker_embedding


class ReshuffleSampler:
    """
    Class to sample items from an iterable in a random order.
    """
    def __init__(self, iterable: Iterable[Any]):
        self.iterable = iterable
        self.num_items = len(iterable)
        self.iterator = self._get_new_iterator()

    def _get_new_iterator(self):
        assert self.num_items > 0, "Iterable must have at least one item."
        indices = np.arange(self.num_items)
        np.random.shuffle(indices)

        return itertools.cycle(indices)

    def sample(self):
        try:
            index = next(self.iterator)
            return self.iterable[index]
        except StopIteration:
            self.iterator = self._get_new_iterator()
            return self.sample()


def eer(y_true, y_score):
    """
    Function computing the Equal Error Rate (EER) from the true labels and predicted scores.
    Args
        y_true: (array-like) True binary labels in range {0, 1} or {-1, 1}.
        y_score: (array-like) Target scores, can either be probability estimates of the positive class,

    Returns: (float) The EER value.

    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)

    return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)


def compute_eer(
    gt_audio_files: List[Path],
    synth_audio_files: List[Path],
    model: EncoderClassifier,
    re_equalize: bool,
    nb_samples: int = 200,
) -> float:
    """
    Computes the Equal Error Rate (EER) between ground truth and synthesized audio files.
    Args:
        gt_audio_files: (List[Path]) List of ground truth audio files.
        synth_audio_files: (List[Path]) List of synthesized audio files.
        model: (EncoderClassifier) The speaker embedding model used to extract embeddings.
        re_equalize: (bool) Whether to re-equalize the synthesized audio files against their GT counterpart.
        nb_samples: (int) Number of samples to use for EER computation. Default is 200.

    Returns: (float) The EER value.

    """
    gt_sampler = ReshuffleSampler(gt_audio_files)
    synth_sampler = ReshuffleSampler(synth_audio_files)

    # create a list of labels
    # we assume that the ASV is trying to detect real sample
    # true labels (gt on gt) are 1, impostors (synth) are 0
    y = np.zeros((2 * nb_samples,))
    y[:nb_samples] = 1  # genuine trials

    y_hat = np.zeros((2 * nb_samples,))

    for i, expected_label in tqdm(
        enumerate(y), desc=f"Computing EER -- re_equalize={re_equalize}"
    ):
        ref_audio_file = gt_sampler.sample()
        gt_audio_file = gt_sampler.sample()
        synth_audio_file = synth_sampler.sample()

        ref_embedding = get_speaker_embedding(model, ref_audio_file)
        gt_embedding = get_speaker_embedding(model, gt_audio_file)
        synth_embedding = (
            get_speaker_embedding(
                model, synth_audio_file, re_equalize_ref=ref_audio_file
            )
            if re_equalize
            else get_speaker_embedding(model, synth_audio_file)
        )

        if expected_label == 0:
            y_hat[i] = cosine(
                ref_embedding, gt_embedding
            )  # gt on gt should yield 1 - 1 = 0
        else:
            y_hat[i] = cosine(
                ref_embedding, synth_embedding
            )  # gt on synth should yield 1 - 0 = 1

    return eer(y, y_hat)
