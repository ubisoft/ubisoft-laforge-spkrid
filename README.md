© [2025] Ubisoft Entertainment. All Rights Reserved

# Analyzing and Improving Speaker Similarity Assessment for Speech Synthesis

[![arXiv](...)](...) # path to arxiv paper

Official repository for [Analyzing and Improving Speaker Similarity Assessment for Speech Synthesis](...).

**Abstract**: Modeling voice identity is challenging due to its multifaceted nature. 
In generative speech systems, identity is often assessed using automatic speaker verification (ASV) embeddings, designed for discrimination rather than characterizing identity.
This paper investigates which aspects of a voice are captured in such representations. We find that widely used ASV embeddings focus mainly on static features like timbre and pitch range, while
neglecting dynamic elements such as rhythm. We also identify confounding factors that compromise speaker similarity measurements and suggest mitigation strategies.
To address these gaps, we propose U3D, a metric that evaluates speakers’ dynamic rhythm patterns. This work contributes to the ongoing challenge of assessing
speaker identity consistency in the context of ever-better voice cloning systems.

## Get Started

### Build Environment From Scratch

This repository is expected to run in Python 3.11. The environment is defined in `requirements.txt`.

To install the dependencies, you can use the following commands:

```
pip install -r requirements.txt
```

Then, move to the project's root directory and run

```
pip install -e .
```

**Note**: The repository uses torchaudio for audio processing, which requires `ffmpeg` to be installed on your system.
You will need to install ffmpeg if you don't already have it.

On linux, you can run the command:
```
sudo apt-get update -y && apt-get upgrade -y && apt-get install ffmpeg
```

### Use Pre-Built Dockerfile

A Dockerfile is also provided for easy setup of the environment.

## Example usage

Examples of how to apply the Equalization Matching trick in EER computation and how to use the U3D metric are shown in `scripts/`.

### Compute EER with Equalization Matching

To compute the Equal Error Rate (EER) with Equalization Matching, run the `scripts/compute_eer.py` script on a set of synthesized and ground truth voice samples.
The script computes and reports the EER both with and without the equalization matching technique. It generates random same-speaker pairs: ground truth vs. ground truth,
and synthesized vs. ground truth. When equalization is enabled, the synthesized file is re-equalized using its ground truth counterpart. Speaker embeddings are
then extracted, and the EER is calculated using cosine distance.

To run the script, use the following command:

```
python scripts/compute_eer.py
```

### Compute U3D Metric

To compute the U3D metric for a set of synthesized voices against a ground truth set, run the `scripts/compute_u3d.py` script.
The script calculates the average Wasserstein distance between the duration distributions of phoneme-like speech unit groups in two settings:

- same_speaker: ground truth vs. synthesized lines from the same speaker

- different_speakers: ground truth lines from one speaker vs. synthesized lines from all other speakers

To run the script, use the following command:

```
python scripts/compute_u3d.py
```

### Configuration

Both scripts can be configured through the `scripts/config.yaml` file, where:

- `ground_truth_dir` and `synth_dir` point respectively to the ground truth and synthesized files directories. If left empty, 
the scripts will use the data found in the directories `scripts/audios/ground_truth` and `scripts/audios/synthesized`. 
- `device` allows the user to run the script on a specific device (to run on GPU, you'll have to [install the GPU version of PyTorch](https://pytorch.org/get-started/locally/)).

For the EER computation script, additional parameters can be set in the `eer` section of the configuration file:
- `embedding_type` specifies the type of speaker embeddings to use (either `ecapa-tdnn` / `resnet-tdnn` / `x-vector`).
- `nb_samples` specifies the number of pairs to sample for the EER computation (default is 200).

The directories should contain audio files in `.wav` format. The name of parent directories containing audio files
is expected to be the speaker ID. The provided synthesized and ground truth directories should have the same structure,
and can have any arbitrary number of speakers and audio files per speaker.

Example:
```
synthesized/
    speaker1/
        audio1.wav
        audio2.wav
    speaker2/
        audio1.wav
        audio2.wav

ground_truth/
    speaker1/
        audio1.wav
        audio2.wav
    speaker2/
        audio1.wav
        audio2.wav
```

#### Data

A basic dataset formed of lines from the [CMU ARCTIC](http://www.festvox.org/cmu_arctic/) dataset is provided in the `scripts/audios/` directory for testing purposes.
Specifically, the first 20 lines from the speakers "aew", "jmk", "lnh".

## Citation

If you found this work helpful please consider citing our paper.

```
...
```

© [2025] Ubisoft Entertainment. All Rights Reserved