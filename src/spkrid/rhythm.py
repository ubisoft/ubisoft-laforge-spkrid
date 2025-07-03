"""
This module provides functionality to extract segment durations from audio files using a HuBERT model and a segmenter.
"""

from itertools import pairwise
from pathlib import Path
from typing import Any, List

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from pandas import DataFrame
from tqdm import tqdm


@torch.inference_mode()
def encode(hubert, audio_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes the audio tensor using the HuBERT model to obtain units and log probabilities.
    Args:
        hubert: (HubertSoft) The HuBERT model.
        audio_tensor: (torch.Tensor) A tensor containing the audio data, shape (batch_size, num_channels, num_samples).

    Returns: (torch.Tensor, torch.Tensor) Units and log probabilities.

    """
    units = hubert.units(audio_tensor)
    logits = hubert.logits(units)
    log_probs = F.log_softmax(logits, dim=-1)

    return units.transpose(1, 2), log_probs


def is_trailing_silence(i: int, sound_name: str, num_segments: int) -> bool:
    """
    Checks if the current segment is a trailing silence segment.
    Args:
        i: (int) The index of the current segment.
        sound_name: (str) The name of the sound segment, e.g., "SILENCE".
        num_segments: (int) The total number of segments in the utterance.

    Returns: (bool) True if the segment is a trailing silence, False otherwise.

    """
    return sound_name == "SILENCE" and (i == 0 or i == num_segments - 1)


def compute_duration(start: int, end: int, hop_length: float = 0.02) -> float:
    """
    Computes the duration of a segment in milliseconds.
    Args:
        start: (int) The start index of the segment.
        end: (int) The end index of the segment.
        hop_length: (float) The hop length in seconds, default is 0.02 seconds (20 ms).

    Returns: (float) The duration of the segment in milliseconds.

    """
    return (end - start) * hop_length * 1000  # convert to milliseconds


def extract_durations_from_utterance(
    audio_file: Path, segments: List[Any], boundaries: List[int]
):
    """
    Extracts durations from the segments of an utterance.
    Args:
        audio_file: (Path) The path to the audio file.
        segments: (List[Any]) A list of segments, where each segment is a named tuple or object with a 'name' attribute.
        boundaries: (List[int]) A list of boundaries indicating the start and end indices of each segment.

    Returns: (List[tuple]) A list of tuples containing the speaker name, group name, and duration of each segment.

    """
    return [
        (audio_file.parent.stem, sound.name.lower(), compute_duration(a, b))
        for i, (sound, (a, b)) in enumerate(zip(segments, pairwise(boundaries)))
        if not is_trailing_silence(i, sound.name, len(segments))
    ]


def extract_durations(hubert, segmenter, dir_path: Path) -> DataFrame:
    """
    Extracts durations of segments for each speaker in the given directory path.
    Args:
        hubert: The HuBERT model used for encoding audio.
        segmenter: The segmenter model used to segment the audio.
        dir_path: (Path) The directory path containing audio files.

    Returns: (DataFrame) A DataFrame containing the speaker name, group name, and duration of each segment.

    """
    duration_data = []
    for audio_file in tqdm(
        list(dir_path.rglob("*.wav")),
        desc=f"Extracting durations for each speaker in {dir_path.name}",
    ):
        audio_tensor, fs = torchaudio.load(audio_file)
        audio_tensor = AF.resample(audio_tensor, fs, 16000)
        audio_tensor = audio_tensor.unsqueeze(0).to(hubert.proj.weight.device)

        _, log_probs = encode(hubert, audio_tensor)
        log_probs = log_probs.squeeze().cpu().numpy()
        segments, boundaries = segmenter(log_probs)
        duration_data.extend(
            extract_durations_from_utterance(audio_file, segments, boundaries)
        )

    duration_data = DataFrame(duration_data, columns=["speaker", "group", "duration"])

    return duration_data
