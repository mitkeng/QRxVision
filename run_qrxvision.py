import os
os.environ['MPLBACKEND'] = 'Agg'
import pandas as pd
import csv
import shutil
import ase
import cirpy
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors
from urllib.request import urlopen
from pubchempy import get_compounds, Compound
import math
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import statistics
import sys
import re
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
from statistics import*
from urllib.parse import quote
from copy import deepcopy
from ase import io
from ase.io import read, write
import __main__
import qrcode
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import glob
import pickle
from collections import Counter
import argparse

# Data constants for descriptor calculation
ATOM_DATA = {
    "H":  {"r": 1.2,  "m": 1.0,   "en": 2.2,  "ie": 13.598, "p": 0.667},
    "C":  {"r": 1.7,  "m": 12.0,  "en": 2.55, "ie": 11.26,  "p": 1.76},
    "N":  {"r": 1.55, "m": 14.0,  "en": 3.04, "ie": 14.534, "p": 1.1},
    "O":  {"r": 1.52, "m": 16.0,  "en": 3.44, "ie": 13.618, "p": 0.802},
    "P":  {"r": 1.8,  "m": 31.0,  "en": 2.19, "ie": 10.487, "p": 3.63},
    "S":  {"r": 1.8,  "m": 32.0,  "en": 2.58, "ie": 10.36,  "p": 2.9},
    "F":  {"r": 1.35, "m": 19.0,  "en": 3.98, "ie": 17.423, "p": 0.557},
    "Cl": {"r": 1.75, "m": 35.34, "en": 3.16, "ie": 12.968, "p": 2.18},
    "Br": {"r": 1.83, "m": 79.9,  "en": 2.96, "ie": 11.8,   "p": 3.05}
}

# Load VGG16 model once (assumed to be available in the environment)
MODEL = VGG16(weights='imagenet', include_top=False, pooling='max')

def barcode_id(smi_, drug_dic):
    for i, k in drug_dic.items():
        if k == smi_:
            return i
    return None # Return None if not found

def get_descriptors(smiles, labels):
    mol = Chem.MolFromSmiles(smiles)
    mol2 = Chem.MolFromSmiles(smiles)
    if not mol: return None

    Chem.Kekulize(mol2) # Kekulize might fail or change the molecule, ensure it's handled

    stats = {"r": 0, "m": 0, "en": 0, "ie": 0, "p": 0, "hetero": 0, "halo": 0, "pos": 0, "neg": 0}
    halogens = {'F', 'Cl', 'Br', 'I'}

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in ATOM_DATA:
            d = ATOM_DATA[sym]
            stats['r'] += d['r']; stats['m'] += d['m']; stats['en'] += d['en']
            stats['ie'] += d['ie']; stats['p'] += d['p']

        if sym not in ['C', 'H']: stats['hetero'] += 1
        if sym in halogens: stats['halo'] += 1
        if atom.GetFormalCharge() > 0: stats['pos'] += 1
        if atom.GetFormalCharge() < 0: stats['neg'] += 1

    num_atoms = mol.GetNumAtoms()
    data = {
        "smile": smiles,
        "radii_mass2": round((stats['r'] * num_atoms) / stats['m'], 2) if stats['m'] else 0,
        "electronic2": round((stats['ie'] * stats['en']) / (stats['p'] * num_atoms), 2) if num_atoms and stats['p'] else 0,
        "double_bond2": len([b for b in mol2.GetBonds() if b.GetBondTypeAsDouble() == 2.0]),
        "triple_bond2": len([b for b in mol2.GetBonds() if b.GetBondTypeAsDouble() == 3.0]),
        "hetero2": stats['hetero'],
        "halogen": stats['halo'],
        "ring2": mol.GetRingInfo().NumRings(),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "VSA": round(Descriptors.VSA_EState2(mol), 2),
        "logP": round(Descriptors.MolLogP(mol), 2),
        "pos_charge": stats['pos'],
        "neg_charge": stats['neg']
    }

    label_keys = "A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,label_list".split(',')
    data.update(dict(zip(label_keys, labels)))

    return data

def enumerate_smiles(smiles, n_variants):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [smiles] * n_variants

    variants = set()
    variants.add(Chem.MolToSmiles(mol, canonical=True))

    attempts = 0
    while len(variants) < n_variants and attempts < 100:
        variants.add(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
        attempts += 1

    res = list(variants)
    while len(res) < n_variants:
        res.append(smiles)
    return res[:n_variants]

def extract_features_batch(img_paths, model):
    images = []
    for img_path in img_paths:
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)
        except FileNotFoundError:
            # print(f"Error: {img_path} not found.") # Suppress printing in function
            continue

    if not images:
        return np.array([])

    images_array = np.array(images)
    images_array = preprocess_input(images_array)
    features = model.predict(images_array)
    return features

def calculate_similarity_batch(features1, features2_list):
    if features2_list.size == 0:
      return np.array([])
    return cosine_similarity([features1], features2_list)[0]

def filter_max_value_duplicates(data):
    max_values = {}
    for item in data:
        string_key = item[1]
        current_value = item[2]

        if string_key not in max_values or current_value > max_values[string_key][2]:
            max_values[string_key] = item

    string_counts = Counter(item[1] for item in data)
    result = [item for string_key, item in max_values.items() if string_counts[string_key] > 1]

    return result

def process_compound_similarity(compound_name, compound_SMILE, num_top_results=10):
    # --- Cleanup previous run files/folders (internal to a single compound processing, keep minimal) ---
    try:
        os.remove("out_try0.csv")
        os.remove("out_try1.csv")
        os.remove("out_try2.csv")
        os.remove("output_with_duplicates.csv")
        os.remove("test_dataset.csv")
    except OSError:
        pass
    # Removed shutil.rmtree("test_QRcode") and shutil.rmtree("Mole_feat3QR") from here

    # --- 1. Load Intrinsic Data and create drug_dic ---
    pd.read_csv("Drug_ML_RE_training.csv") # This line is redundant if smiles_df is not used

    drug_dic = {}
    try:
        with open('Cancer_drug3.csv', 'r') as DR2:
            next(DR2) # Skip header
            for d2 in DR2:
                try:
                    parts = d2.strip().split(',')
                    if len(parts) > 1:
                        key = parts[0].strip().replace("'", '').replace('"', '') # FIXED LINE
                        value = parts[1]
                        drug_dic[key] = value
                except IndexError:
                    pass
    except FileNotFoundError:
        print("Error: Cancer_drug3.csv not found.")
        return []


    # --- 2. Generate QR codes for training data (Mole_feat3QR) ---
    # Create the main Mole_feat3QR directory if it doesn't exist already
    try:
        os.mkdir("Mole_feat3QR")
    except FileExistsError:
        pass

    try:
        with open('Drug_ML_RE_training.csv', 'r') as DR:
            next(DR) # Skip header
            s = 0
            for dr_line in DR:
                s += 1
                br = dr_line.strip().replace(' ', '').split(',')[:-1]
                smi_ = dr_line.strip().replace(' ', '').split(',')[0]
                data = "*".join(br)

                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4,
                )
                writer_options = {'write_text': False}
                try:
                    qr.add_data(data)
                    qr.make(fit=True)
                    img = qr.make_image(fill_color="black", back_color="white")
                    fil_nam = barcode_id(smi_, drug_dic)
                    if fil_nam: # Ensure a valid filename is found
                        img.save(f"Mole_feat3QR/{fil_nam}.png", options=writer_options)
                except Exception:
                    pass
    except FileNotFoundError:
        print("Error: Drug_ML_RE_training.csv not found.")
        return []


    # --- 3. User input and test data preparation ---
    test_SMILE = compound_SMILE
    df_abnTOTAL = pd.read_csv('abnTOTAL_dataset3.csv')

    num_rows_to_duplicate = 0 # As per original code, not duplicating for test input
    rows_to_duplicate = df_abnTOTAL.head(num_rows_to_duplicate)
    duplicated_df = pd.concat([df_abnTOTAL, rows_to_duplicate], ignore_index=False)
    output_csv_file = 'output_with_duplicates.csv'
    duplicated_df.to_csv(output_csv_file, mode='w', header=True, index=False) # Use 'w' to overwrite

    dff = pd.read_csv('output_with_duplicates.csv')

    dff2 = dff[dff.columns[13:]].copy() # Use .copy() to avoid SettingWithCopyWarning
    dff2.insert(0, column='smile', value=test_SMILE)
    # The original loop `for d in range(len(dff)): dff2['smile']= test_SMILE` is redundant

    dff2.to_csv('out_try0.csv', mode='w', header=True, index=False)

    with open('out_try0.csv', 'r') as inp:
        lines = inp.readlines()

    with open('out_try1.csv', 'w') as out:
        for line in lines:
            if not 'label_list' in line:
                out.write(line)

    with open('out_try0.csv', newline='') as file:
        reader = csv.reader(file, delimiter = ' ')
        headings = next(reader)

    head_list = []
    for he in headings:
        head_list.append(he.split(","))

    df5 = pd.read_csv("out_try1.csv", header=None, names=head_list[0])
    df5.to_csv("out_try2.csv", index=False)

    # --- 4. Run Reference Compound Scan (Feature Extraction for test compound) ---
    file_name = "out_try2.csv"

    smi_list = []
    label_list = []

    try:
        with open(file_name, encoding='utf-8-sig') as smile_file:
            first_line = True
            for fi in smile_file:
                if first_line:
                    first_line = False
                    continue # Skip header

                parts = fi.strip().split(",")
                if len(parts) > 0:
                    smi_list.append(parts[0].strip())
                    label_list.append(parts[1:])
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        return []

    # --- 5. Generate descriptors and augment SMILES for test compound ---
    output_file_test_dataset = "test_dataset.csv"
    results = []

    for sm, lbls in zip(smi_list, label_list):
        desc = get_descriptors(sm, lbls)
        if desc: results.append(desc)

    pd.DataFrame(results).to_csv(output_file_test_dataset, index=False)

    df_test = pd.read_csv(output_file_test_dataset)
    smiles_col = 'smile'
    n_aug = 1
    augmented_rows = []

    for _, row in df_test.iterrows():
        variants = enumerate_smiles(row[smiles_col], n_variants=n_aug)

        for v in variants:
            new_row = row.copy()
            new_row[smiles_col] = v
            augmented_rows.append(new_row)

    df_augmented = pd.DataFrame(augmented_rows)
    df_augmented.to_csv(output_file_test_dataset, index=False)

    # --- 6. Generate QR codes for test data (test_QRcode) ---
    # Create the main test_QRcode directory if it doesn't exist already
    base_test_qr_dir = "test_QRcode"
    os.makedirs(base_test_qr_dir, exist_ok=True)

    # Create a unique subdirectory for the current compound
    sanitized_compound_name = re.sub(r'[\/*?:"<>|]', '', compound_name).replace(' ', '_')
    current_compound_qr_dir = os.path.join(base_test_qr_dir, sanitized_compound_name)
    os.makedirs(current_compound_qr_dir, exist_ok=True) # Ensure this specific directory exists

    try:
        with open(output_file_test_dataset, 'r') as DR_test:
            next(DR_test) # Skip header
            s = 0
            for dr_line in DR_test:
                s += 1
                br = dr_line.strip().replace(' ', '').split(',')[:-1]
                smi_ = dr_line.strip().replace(' ', '').split(',')[0]
                data = "*".join(br)

                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4,
                )
                writer_options = {'write_text': False}
                try:
                    qr.add_data(data)
                    qr.make(fit=True)
                    img = qr.make_image(fill_color="black", back_color="white")
                    # For test data, we don't have an ID from drug_dic, use a simple counter
                    test_qr_filename = os.path.join(current_compound_qr_dir, f"test_{s}.png")
                    img.save(test_qr_filename, options=writer_options)
                except Exception as e:
                    print(f"Error generating QR code for {smi_} in {current_compound_qr_dir}: {e}")
    except FileNotFoundError:
        print(f"Error: {output_file_test_dataset} not found.")
        return []


    # --- 7. Feature extraction from QR codes and similarity calculation ---
    mole_dir = '/content/Mole_feat3QR'
    # IMPORTANT: test_dir should now point to the specific compound's QR code directory
    test_dir = current_compound_qr_dir

    mole_paths = glob.glob(os.path.join(mole_dir, '*'))
    test_paths = glob.glob(os.path.join(test_dir, '*'))

    features_cache_file = 'mole_features_cache.pkl'

    mole_features = np.array([])
    if os.path.exists(features_cache_file):
        with open(features_cache_file, 'rb') as f:
            mole_features = pickle.load(f)
    else:
        mole_features = extract_features_batch(mole_paths, MODEL)
        if mole_features.size > 0:
            with open(features_cache_file, 'wb') as f:
                pickle.dump(mole_features, f)

    pair_likeness2 = []

    test_features = extract_features_batch(test_paths, MODEL)
    test_filenames = [os.path.basename(p) for p in test_paths]
    mole_filenames = [os.path.basename(p) for p in mole_paths]

    if test_features.size > 0 and mole_features.size > 0:
        for i, test_feat in enumerate(test_features):
            similarity_scores = calculate_similarity_batch(test_feat, mole_features)

            for j, score in enumerate(similarity_scores):
                pair_likeness2.append((test_filenames[i], mole_filenames[j], score))

    # --- 8. Filtering and displaying results ---
    filtered_list = filter_max_value_duplicates(pair_likeness2)
    top_n_similar_compounds = sorted(filtered_list, key=lambda x: x[2], reverse=True)[:num_top_results]

    return top_n_similar_compounds


# Example Usage (for demonstration in Colab or when running the .py file)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process chemical compounds for similarity analysis.')
    parser.add_argument('--smile', type=str, help='A single SMILE string to process.')
    parser.add_argument('--name', type=str, help='Name for the single SMILE compound (optional).')
    parser.add_argument('--csv_file', type=str, help='Path to a CSV file containing SMILES strings for batch processing. Expected columns: "name", "smile".')
    parser.add_argument('--output_file', type=str, help='Path to an output CSV file to save the results.')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top similar compounds to display/save (default: 10).')
    args = parser.parse_args()

    # --- Global cleanup of QR code directories ---
    try:
        shutil.rmtree("test_QRcode")
    except OSError:
        pass
    try:
        shutil.rmtree("Mole_feat3QR")
    except OSError:
        pass
    # --- End Global cleanup ---

    # Create dummy files for demonstration if they don't exist
    if not os.path.exists('Drug_ML_RE_training.csv'):
        pd.DataFrame([{'SMILES': 'CC(=O)Oc1ccccc1C(=O)O', 'Label1': 'A', 'Label2': 'R'}]).to_csv('Drug_ML_RE_training.csv', index=False)
    if not os.path.exists('Cancer_drug3.csv'):
        with open('Cancer_drug3.csv', 'w') as f:
            f.write('CompoundID,SMILES\n')
            f.write('Aspirin,CC(=O)Oc1ccccc1C(=O)O\n')
            f.write('Paracetamol,CC(=O)Nc1ccc(O)cc1\n')
    if not os.path.exists('abnTOTAL_dataset3.csv'):
        dummy_data = {'col' + str(i): [0] for i in range(1, 15)}
        dummy_data['smile'] = ['dummy_smile']
        pd.DataFrame(dummy_data).to_csv('abnTOTAL_dataset3.csv', index=False)

    if args.smile and args.csv_file:
        print("Error: Cannot specify both --smile and --csv_file. Choose one.")
        sys.exit(1)
    elif args.smile:
        compound_name = args.name if args.name else "Unknown Compound"
        compound_SMILE = args.smile
        print(f"--- Running process_compound_similarity for single compound: {compound_name} ({compound_SMILE}) ---")
        try:
            top_similar_compounds = process_compound_similarity(compound_name, compound_SMILE, num_top_results=args.top_n)
            print(f"Top {args.top_n} Similar Compounds:")
            for item in top_similar_compounds:
                print(f"Test: {item[0]}, Reference: {item[1].replace('.png','')}, Similarity: {item[2]:.4f}")

            if args.output_file:
                output_data = []
                for item in top_similar_compounds:
                    output_data.append({
                        'TestCompoundName': compound_name,
                        'TestCompoundSMILE': compound_SMILE,
                        'TestImage': item[0],
                        'ReferenceCompound': item[1].replace('.png', ''),
                        'SimilarityScore': item[2]
                    })
                output_df = pd.DataFrame(output_data)
                output_df.to_csv(args.output_file, index=False)
                print(f"Results for '{compound_name}' saved to {args.output_file}")

        except Exception as e:
            print(f"An error occurred during processing: {e}")
    elif args.csv_file:
        print(f"--- Running process_compound_similarity for compounds from CSV: {args.csv_file} ---")
        try:
            df_smiles = pd.read_csv(args.csv_file)
            if 'smile' not in df_smiles.columns:
                print("Error: CSV file must contain a 'smile' column.")
                sys.exit(1)

            all_compound_results_for_printing = [] # To store structure like before for printing
            final_output_list_for_csv = [] # To store flattened data for CSV

            for index, row in df_smiles.iterrows():
                compound_SMILE = row['smile']
                compound_name = row['name'] if 'name' in row and pd.notna(row['name']) else compound_SMILE
                print(f"\nProcessing {compound_name} ({compound_SMILE})...")
                results_for_compound = process_compound_similarity(compound_name, compound_SMILE, num_top_results=args.top_n)

                all_compound_results_for_printing.append({'compound_name': compound_name, 'compound_SMILE': compound_SMILE, 'similar_compounds': results_for_compound})

                for item in results_for_compound:
                    final_output_list_for_csv.append({
                        'TestCompoundName': compound_name,
                        'TestCompoundSMILE': compound_SMILE,
                        'TestImage': item[0],
                        'ReferenceCompound': item[1].replace('.png', ''),
                        'SimilarityScore': item[2]
                    })

            print(f"\n--- Batch Processing Complete (Top {args.top_n} results per compound) ---")
            for res in all_compound_results_for_printing:
                print(f"\nResults for {res['compound_name']} ({res['compound_SMILE']}):")
                if res['similar_compounds']:
                    print(f"Top {args.top_n} Similar Compounds:")
                    for item in res['similar_compounds']:
                        print(f"  Test: {item[0]}, Reference: {item[1].replace('.png','')}, Similarity: {item[2]:.4f}")
                else:
                    print("  No similar compounds found or an error occurred.")

            if args.output_file:
                output_df = pd.DataFrame(final_output_list_for_csv)
                output_df.to_csv(args.output_file, index=False)
                print(f"Batch results saved to {args.output_file}")

        except FileNotFoundError:
            print(f"Error: CSV file not found at {args.csv_file}")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred during CSV processing: {e}")
    else:
        print("No input provided. Use --smile <SMILE_STRING> or --csv_file <PATH_TO_CSV>.")
        sys.exit(1)