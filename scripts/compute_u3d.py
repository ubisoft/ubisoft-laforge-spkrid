"""
Entrypoint script to measure the U3D metric on a set of ground truth and synthesized audio files.
"""

import itertools
from pathlib import Path

import pandas
import torch
import yaml
from pandas import DataFrame
from scipy.stats import wasserstein_distance

from spkrid.rhythm import extract_durations

script_dir = Path(__file__).parent.resolve()


def main() -> DataFrame:
    config_file = script_dir / "config.yaml"
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config["ground_truth_dir"]:
        ground_truth_dir = script_dir / "audios" / "ground_truth"
    else:
        ground_truth_dir = Path(config["ground_truth_dir"])

    if not config["synthesis_dir"]:
        synthesis_dir = script_dir / "audios" / "synthesized"
    else:
        synthesis_dir = Path(config["synthesis_dir"])

    device = config["device"]

    print("Loading HuBERT model...")
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
    hubert = hubert.to(device)

    # use the URhythmic segmenter with 8 clusters grouped into Approximant, Fricative, Nasal, Silence, Stop, and Vowel
    print("Loading URhythmic segmenter...")
    segmenter = torch.hub.load("bshall/urhythmic:main", "segmenter", num_clusters=8)

    gt_durations = extract_durations(hubert, segmenter, ground_truth_dir)
    synth_durations = extract_durations(hubert, segmenter, synthesis_dir)

    gt_groups = gt_durations.groupby(["speaker", "group"])
    synth_groups = synth_durations.groupby(["speaker", "group"])

    records = []
    for (gt_key, gt_df), (synth_key, synth_df) in itertools.product(
        gt_groups, synth_groups
    ):
        gt_speaker, gt_group = gt_key
        synth_speaker, synth_group = synth_key

        if gt_group != synth_group:
            continue

        if gt_df.empty or synth_df.empty:
            continue

        dist = wasserstein_distance(gt_df.duration, synth_df.duration)

        records.append(
            {
                "gt_speaker": gt_speaker,
                "synth_speaker": synth_speaker,
                "group": gt_group,
                "dist": dist,
                "same_speaker": gt_speaker == synth_speaker,
            }
        )

    distances = DataFrame(records)
    mean_distances = (
        distances.groupby(["group", "same_speaker"])["dist"].mean().reset_index()
    )
    mean_distances = mean_distances.pivot(
        index="group", columns="same_speaker", values="dist"
    )
    mean_distances.columns = ["different_speaker", "same_speaker"]
    mean_distances = mean_distances.reset_index()

    overall_means = mean_distances[["same_speaker", "different_speaker"]].mean()
    overall_row = {
        "group": "average",
        "same_speaker": overall_means["same_speaker"],
        "different_speaker": overall_means["different_speaker"],
    }
    mean_distances = pandas.concat(
        [mean_distances, DataFrame([overall_row])], ignore_index=True
    )

    mean_distances[["same_speaker", "different_speaker"]] = mean_distances[
        ["same_speaker", "different_speaker"]
    ].round(2)

    return mean_distances


if __name__ == "__main__":
    try:
        mean_distances = main()
        print(mean_distances.to_string(index=False))
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
