import torch
from datasets import load_dataset, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import evaluate
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import json
import librosa
import os
from pathlib import Path
from huggingface_hub import login
import json
import glob
import re, pickle
from collections import defaultdict
import evaluate


def run_inference(model_data, checkpoint, device, fleurs):
    hf_token = ""
    login(hf_token)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(checkpoint)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        checkpoint,  # Path to your checkpoint
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    model.to(device)

    results = []

    for item in tqdm(fleurs, desc="Transcribing audio files", total=len(fleurs)):
        dir_path, file_name = os.path.split(item['path'])
        audio_path = Path(dir_path) / item['audio']['path']

        if audio_path.exists():
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=audio_array, sr=sample_rate)

            if duration < 30:
                # Extract input features
                input_features = processor.feature_extractor(
                    audio_array, 
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features

                # Move input features to GPU (if available) and set dtype
                input_features = input_features.to(device).to(torch_dtype)

                # Run inference with no gradient tracking
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_features,
                        # language=language,
                        task="transcribe"
                    )

                # Decode to get transcription
                transcription = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                # Append the results
                results.append({
                    'audio': file_name,
                    'ref_tgt_text': item['raw_transcription'],  # Replace with the correct reference text source
                    'pred_tgt_text': transcription
                })

    results_df = pd.DataFrame(results)
    path = f'whisper_results_fleurs_{model_data}.csv'
    results_df.to_csv(path, sep='\t', index=False)

    return path

def clean_whitespaces(text):
    text = re.sub(r"\s+", " ", text)  # replace any successive whitespaces with a space
    return text

def clean_text(text, target=None):
    text = text.strip()
    if target == 'eng': pattern = re.compile(r"[^\w\s'?,.!]|_", re.UNICODE)
    else:
        pattern = re.compile(r"[^\w\s?,.!]|_", re.UNICODE)
    # Substitute matched characters with an empty string
    text = re.sub(pattern, ' ', text)
    text = text.replace("_", "")
    text = clean_whitespaces(text).strip()
    return text

def clean_punctuation(text, target=None):
    # This regex pattern matches all characters that are not words, numbers, whitespace, hyphens, or apostrophes '-
    text = text.strip()
    if target == 'eng': pattern = re.compile(r"[^\w\s]|_", re.UNICODE)
    else:
        pattern = re.compile(r"[^\w\s]|_", re.UNICODE)
    # Substitute matched characters with an empty string
    text = re.sub(pattern, ' ', text)
    text = clean_whitespaces(text).strip()
    return text

def get_data(txt_file, lower=False, lang=None, to_clean = False, clean_punct = False):
    refs, preds = [], []
    with open(txt_file) as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            if len(line) != 3:
                ref = line[0]
                pred = ''
                print('err', line)
            else: 
                _id, ref, pred = line
                if lower:
                    ref = ref.lower()
                    pred = pred.lower()
                if clean_punct:
                    ref = clean_punctuation(ref, target=lang)
                    pred = clean_punctuation(pred, target=lang)
                if to_clean:
                    ref = clean_text(ref, target=lang)
                    pred = clean_text(pred, target=lang)                    
            refs.append(ref)
            preds.append(pred)
    return refs, preds

def calculate_wer(txt_file):
    wer = evaluate.load("wer")

    if os.path.exists(txt_file):
        ## Normalized - text is lowercased, and all punctuation is removed
        print('Normalized results')
        refs, preds = get_data(txt_file, lower=True, lang="kaz",clean_punct=True, to_clean=True)
        wer_score = wer.compute(predictions=preds, references=refs)
        print(len(refs), txt_file)
        print('WER', round(wer_score * 100, 2))
        ## Un-Normalized - text casing is preserved and basic punctuation is preserved (?!,.')
        print('Un-Normalized results')
        refs, preds = get_data(txt_file, lower=False, lang="kaz", to_clean=True, clean_punct=False)
        wer_score = wer.compute(predictions=preds, references=refs)
        print(len(refs), txt_file)
        print('WER', round(wer_score * 100, 2))




model_name = "whisper_3_3"
# checkpoint = "/scratch/rakhat_meiramov/ksc2-crowd-new/final"
# checkpoint = "/scratch/rakhat_meiramov/base-kaznerd-new/final"
checkpoint = "issai/whisper-turbo"

device = "cuda:0"
print(f"{model_name} checkpoint:")
print("Loading FLEURS")
fleurs = load_dataset("google/fleurs", "kk_kz", split="test", trust_remote_code=True)

print("Running inference")
txt_path = run_inference(model_name, checkpoint, device, fleurs)

print("Evaluating")
calculate_wer(txt_path)








        


