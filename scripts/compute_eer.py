"""
Entrypoint script to compute Equal Error Rate (EER) with equalization matching
"""

import os
from pathlib import Path

import yaml
from pandas import DataFrame

from spkrid.eer import compute_eer
from spkrid.speaker_embedding import load_speaker_embedding_model

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
    nb_samples = config["eer"]["nb_samples"]
    embedding_type = config["eer"]["embedding_type"]

    print("Loading Speaker Embedding Model...")
    model = load_speaker_embedding_model(embedding_type=embedding_type, device=device)

    # we only compute EER on speakers that have both ground-truth and synthesized directories
    gt_speakers = [f.stem for f in ground_truth_dir.glob("*") if os.path.isdir(f)]
    synth_speakers = [f.stem for f in synthesis_dir.glob("*") if os.path.isdir(f)]
    common_speakers = list(set(gt_speakers) & set(synth_speakers))

    records = []
    for speaker in common_speakers:
        print(f"Processing speaker: {speaker}")

        gt_spkr_dir = ground_truth_dir / speaker
        synth_spkr_dir = synthesis_dir / speaker

        gt_audio_files = list(gt_spkr_dir.glob("*.wav"))
        synth_audio_files = list(synth_spkr_dir.glob("*.wav"))

        eer_raw = compute_eer(
            gt_audio_files,
            synth_audio_files,
            model,
            re_equalize=False,
            nb_samples=nb_samples,
        )
        eer_re_eq = compute_eer(
            gt_audio_files,
            synth_audio_files,
            model,
            re_equalize=True,
            nb_samples=nb_samples,
        )

        records.append({"speaker": speaker, "eer": eer_raw, "is_re_equalized": False})
        records.append({"speaker": speaker, "eer": eer_re_eq, "is_re_equalized": True})

    eer_scores = DataFrame(records)
    eer_scores = eer_scores.pivot(
        index="speaker", columns="is_re_equalized", values="eer"
    )
    eer_scores.columns = ["raw", "re_equalized"]
    eer_scores = eer_scores.reset_index()

    eer_scores[["raw", "re_equalized"]] = eer_scores[["raw", "re_equalized"]].round(2)

    return eer_scores


if __name__ == "__main__":
    try:
        eer_scores = main()
        print(eer_scores.to_string(index=False))
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
