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

def calculate_cer(txt_file):
    cer = evaluate.load("cer")

    if os.path.exists(txt_file):
        ## Normalized - text is lowercased, and all punctuation is removed
        print('Normalized results')
        refs, preds = get_data(txt_file, lower=True, lang="kaz",clean_punct=True, to_clean=True)
        wer_score = cer.compute(predictions=preds, references=refs)
        print(len(refs), txt_file)
        print('CER', round(wer_score * 100, 2))
        ## Un-Normalized - text casing is preserved and basic punctuation is preserved (?!,.')
        print('Un-Normalized results')
        refs, preds = get_data(txt_file, lower=False, lang="kaz", to_clean=True, clean_punct=False)
        wer_score = cer.compute(predictions=preds, references=refs)
        print(len(refs), txt_file)
        print('CER', round(wer_score * 100, 2))

def main(csv_folder):
    # Collect and concatenate all CSVs into a single DataFrame
    filenames = glob.glob(os.path.join(csv_folder, "*.csv"))
    all_data = []

    for filename in filenames:
        df = pd.read_csv(filename, sep='\t', names=["id", "ref", "pred"], skiprows=1)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save the combined data as a temporary CSV file
    temp_csv_file = os.path.join(csv_folder, "combined_temp.csv")
    combined_df.to_csv(temp_csv_file, sep='\t', index=False, header=False)

    # Pass the temporary CSV to the CER calculation function
    calculate_cer(temp_csv_file)

    # Optional: Clean up temporary file
    os.remove(temp_csv_file)

if __name__ == "__main__":
    csv_folder = "/home/rakhat_meiramov/asr/kaznerd/test/human/ptsynth_baseline+(ksc+kaznerd)"
    main(csv_folder)