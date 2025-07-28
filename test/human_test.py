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


def run_inference(model_data, checkpoint, device, kaznerd_files):

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(checkpoint)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        checkpoint,  # Path to your checkpoint
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    model.to(device)
    paths = []

    for file in kaznerd_files:
        results = []
        base_path = '/data/rakhat_meiramov/kaznerd_human/recordings/' + os.path.basename(file)[:-4]
        try:
            df = pd.read_csv(file, low_memory=False)
        except Exception as e:
            print(f"Error loading csv: {e}")
            return None, None

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file}"):
            sentence_id = row['sentence_id']
            audio_path = Path(base_path + '/' + sentence_id + ".wav")
            if audio_path.exists():
                audio_array, sample_rate = librosa.load(audio_path, sr=16000)
                duration = librosa.get_duration(y=audio_array, sr=sample_rate)
                if duration < 30:
                    input_features = processor.feature_extractor(
                        audio_array, 
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_features
        
                    input_features = input_features.to(device).to(torch_dtype)
        
                    with torch.no_grad():
                        generated_ids = model.generate(
                            input_features,
                            # language=language,
                            task="transcribe"
                        )
                    
                    transcription = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]
        
                    results.append({
                        'audio': sentence_id,
                        'ref_tgt_text': row['text'],
                        'pred_tgt_text': transcription
                    })

        results_df = pd.DataFrame(results)
        path = f'human/whisper_results_human_{model_data}_{os.path.basename(file)}'
        results_df.to_csv(path, sep='\t', index=False)
        paths.append(path)

    return paths

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




model_name = "ptsynth_baseline+(ksc+kaznerd)"
# checkpoint = "/scratch/rakhat_meiramov/ksc2-crowd-new/final"
# checkpoint = "/scratch/rakhat_meiramov/base-kaznerd-new/final"
checkpoint = "/scratch/rakhat_meiramov/pt_synth_ksc_kaznerd/final"
# checkpoint = "/scratch/rakhat_meiramov/mixed_ksc_kaznerd_new/final"
# checkpoint = "/scratch/rakhat_meiramov/mixed_ksc_kaznerd_1_2/final"

device = "cuda:0"
print(f"{model_name} checkpoint:")
print("Loading KazNERD_Human")
csv_folder = "/data/rakhat_meiramov/kaznerd_human/samples"
kaznerd_files = glob.glob(os.path.join(csv_folder, "*.csv"))

print("Running inference")
txt_paths = run_inference(model_name, checkpoint, device, kaznerd_files)

print("Evaluating")
for txt_path in txt_paths:
    calculate_wer(txt_path)








        


