# KazNERD_ASR

Datasets and evaluation scripts for the paper **"Improved Kazakh Named Entity Transcription Using Synthetic Speech"** (UBMK 2025).

[[Paper]](https://doi.org/10.1109/UBMK67458.2025.11242232)

## Datasets

All datasets are hosted on Hugging Face. Transcripts are based on the [KazNERD](https://github.com/IS2AI/KazNERD) corpus (Kazakh television news text rich in named entities). `sentence_id` corresponds to the KazNERD sentence ID without its trailing `AID` (e.g. `AID343740` ↔ `AID343740AID`).

| Dataset | Type | Splits | Utterances | Size |
|---|---|---|---|---|
| [issai/KazNERD_H](https://huggingface.co/datasets/issai/KazNERD_H) | Human-recorded speech | `sample1`–`sample6` | 531 | 0.1 GB |
| [issai/kaznerd1](https://huggingface.co/datasets/issai/kaznerd1) | Synthetic speech | `train` / `valid` / `test` | 39,493 / 4,908 / 4,385 | 10.8 GB |
| [issai/kaznerd2](https://huggingface.co/datasets/issai/kaznerd2) | Synthetic speech | `train` | 35,609 | 7.7 GB |

- **KazNERD_H**: KazNERD test sentences read aloud by human speakers, used as the real-speech test set.
- **kaznerd1**: synthetic speech of the original KazNERD sentences; its splits follow the KazNERD train/validation/test splits.
- **kaznerd2**: synthetic speech of modified KazNERD training sentences (the transcripts differ from the original KazNERD text), used as additional training data.

Synthetic audio was generated with Kazakh TTS models; the model and speaker for each utterance are given in the `tts_model` and `speaker_id` fields.

Audio is 16 kHz mono WAV, stored in Parquet. Fields:

- `audio`: audio (WAV)
- `text`: Kazakh transcript
- `sentence_id`: KazNERD sentence ID
- `speaker_id`, `gender`, `tts_model`: synthetic datasets only

### Loading

```python
from datasets import load_dataset

human = load_dataset("issai/KazNERD_H")            # splits: sample1 ... sample6
kaznerd1 = load_dataset("issai/kaznerd1")          # splits: train, valid, test
kaznerd2 = load_dataset("issai/kaznerd2", split="train")

sample = kaznerd1["train"][0]
print(sample["text"], sample["tts_model"], sample["speaker_id"])
```

## Evaluation scripts

The `test/` folder contains the scripts used to evaluate fine-tuned Whisper checkpoints. Each script transcribes a test set and reports WER in two settings: normalized (lowercased, punctuation removed) and un-normalized (casing and basic punctuation kept).

| Script | Test set |
|---|---|
| `test/human_test.py` | KazNERD_H (human-recorded speech) |
| `test/kaznerd_test.py` | Synthetic KazNERD test set (`kaznerd1` `test` split) |
| `test/ksc_test.py` | KSC2 crowdsourced test set |
| `test/fleurs_test.py` | FLEURS Kazakh (`kk_kz`) test set |
| `test/find_cer.py` | CER over a folder of prediction files |

Checkpoint and data paths at the bottom of each script are set to the original experiment environment; change them to your local paths before running.

## Citation

```bibtex
@inproceedings{meiramov2025kaznerdasr,
  author    = {Meiramov, Rakhat and Varol, Huseyin Atakan},
  title     = {Improved Kazakh Named Entity Transcription Using Synthetic Speech},
  booktitle = {2025 10th International Conference on Computer Science and Engineering (UBMK)},
  pages     = {1629--1633},
  year      = {2025},
  doi       = {10.1109/UBMK67458.2025.11242232}
}
```

Please also cite [KazNERD](https://aclanthology.org/2022.lrec-1.44/), the source of the transcripts.
