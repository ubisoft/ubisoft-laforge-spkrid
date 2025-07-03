"""
This module provides functionality to load a speaker embedding model and extract speaker embeddings from audio files.
"""

from pathlib import Path
from typing import Optional

import torch
import torchaudio
import torchaudio.functional as AF
from speechbrain.inference.speaker import EncoderClassifier

from spkrid.equalization import filter_welch


def load_speaker_embedding_model(embedding_type: str, device: str) -> EncoderClassifier:
    """
    Loads a speaker embedding model based on the specified embedding type.
    Args:
        embedding_type: (str) The type of embedding model to load. Supported types are:
            - "ecapa-tdnn"
            - "resnet-tdnn"
            - "x-vector"
        device: (str) The device to load the model on, e.g., "cpu" or "cuda".

    Returns: (EncoderClassifier) The loaded speaker embedding model.

    """
    if embedding_type == "ecapa-tdnn":
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}
        )
    elif embedding_type == "resnet-tdnn":
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-resnet-voxceleb", run_opts={"device": device}
        )
    elif embedding_type == "x-vector":
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": device}
        )
    else:
        raise NotImplementedError(f"Embedding type {embedding_type} is not supported.")

    model.eval()

    return model


@torch.inference_mode()
def get_speaker_embedding(
    model: EncoderClassifier,
    audio_file: Path,
    re_equalize_ref: Optional[Path] = None,
) -> torch.Tensor:
    """
    Extracts speaker embedding from an audio file using the specified model.
    Args:
        model: (EncoderClassifier) The speaker embedding model used to extract embeddings.
        audio_file: (Path) The path to the audio file from which to extract the embedding.
        re_equalize_ref: (Optional[Path]) If provided, the audio file to re-equalize the input audio against.

    Returns: (torch.Tensor) The extracted speaker embedding tensor.

    """
    audio_tensor, fs = torchaudio.load(audio_file)
    audio_tensor = AF.resample(audio_tensor, fs, 16000)
    audio_tensor = audio_tensor.to(model.device)

    if re_equalize_ref:
        ref_audio_tensor, fs = torchaudio.load(re_equalize_ref)
        ref_audio_tensor = AF.resample(ref_audio_tensor, fs, 16000)
        audio_tensor, _ = filter_welch(
            ref_audio_tensor.squeeze().cpu().numpy(),
            audio_tensor.squeeze().cpu().numpy(),
        )
        audio_tensor = torch.from_numpy(audio_tensor.copy()).to(model.device)

    if isinstance(model, EncoderClassifier):
        embedding = model.encode_batch(audio_tensor)
    else:
        raise NotImplementedError(
            f"Model type {type(model)} is not supported for speaker embedding extraction."
        )

    return embedding.squeeze().cpu()
