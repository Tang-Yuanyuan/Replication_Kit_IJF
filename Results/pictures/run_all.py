# ====================================================================
# IJF Reproducibility Master Script
# Project: Designing Cost-effective Climate Policies
# ====================================================================

import os
import sys
import optuna
import pickle
import shutil
import subprocess
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

RANDOM_SEED = 120

# ====================================================================
# STEP 0: Environment Setup
# ====================================================================

# ── 1. Resolve Root Directory ──────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 2. Standard Output Directories ─────────────────────────────────────
output_path_table   = os.path.join(ROOT_DIR, 'Results', 'tables')
output_path_picture = os.path.join(ROOT_DIR, 'Results', 'pictures')
output_path_data    = os.path.join(ROOT_DIR, 'Data', 'Intermediate')

raw_data_path = os.path.join(ROOT_DIR, 'Data')
weighted_data_path = os.path.join(ROOT_DIR, 'Data', 'Intermediate')

for _dir in [output_path_table, output_path_picture, output_path_data]:
    os.makedirs(_dir, exist_ok=True)

# ── 3. Configure R Path (Interactive) ──────────────────────────────────
def get_r_path():
    # Attempt 1: Automatically search system environment variables
    print("Attempt to search system environment variables of R...")
    r_path = shutil.which("Rscript")
    
    if r_path:
        print(f"[*] Detected Rscript at: {r_path}")
        confirm = input("Use this path? (Y/n): ").strip().lower()
        if confirm == 'n':
            r_path = None

    # Attempt 2: If automatic detection fails or the user declines, request manual input.
    if not r_path:
        print("\n[!] Rscript could not be detected automatically or was rejected.")
        print("Example path (Windows): C:\\Program Files\\R\\R-4.x.x\\bin\\Rscript.exe")
        print("Example path (Mac/Linux): /usr/local/bin/Rscript")
        r_path = input(">>> Please enter the full path to your 'Rscript' executable: ").strip()
        r_path = r_path.replace('"', '').replace("'", "")

    if not os.path.exists(r_path):
        print(f"ERROR: The path '{r_path}' does not exist. Please restart the script.")
        sys.exit(1)
    
    return r_path

R_EXE = get_r_path()

def run_r_script(script_path):
    """Generic R Script Caller"""
    script_name = os.path.basename(script_path)
    print(f"\n[Running R] {script_name}...")
    try:
        # Using `subprocess` to actually invoke the R environment.
        result = subprocess.run([R_EXE, script_path], 
                                capture_output=True, text=True, check=True)
        print(f"[Success] {script_name} finished. ")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] ERROR in {script_name}:")
        print(e.stderr)
        sys.exit(1)

# ====================================================================
# STEP 1: Section 4.1 - Main Results & Brant Test & Robustness Check
# ====================================================================
print("\n" + "="*50)
print("Step 1: Running Main Analysis of Section 4.1 (R Analysis)")
print("="*50)

path_4_1 = os.path.join(ROOT_DIR, 'Code', 'section4_1', '01_Main_Section_4_1.R')
run_r_script(path_4_1)

# ====================================================================
# STEP 2: Appendix F & G - Robustness Checks
# ====================================================================
print("\n" + "="*50)
print("Step 2: Running Robustness Checks of Section 4.1 (R Analysis)")
print("="*50)

path_4_1_robust = os.path.join(ROOT_DIR, 'Code', 'section4_1', '02_Appendix_F_G_4.1.R')

if os.path.exists(path_4_1_robust):
    run_r_script(path_4_1_robust)
else:
    print(f"\n[!] WARNING: Robustness check script not found at: {path_4_1_robust}")
    print("[!] Skipping robustness checks...")

# ====================================================================
# STEP 3: Section 4.2 - Main Results
# ====================================================================

print("\n" + "="*50)
print("Step 3: Running Main Analysis of Section 4.2 (Python Analysis)")
print("="*50)

path_4_2 = os.path.join(ROOT_DIR, 'Code', 'Section4_2', '03_Main_Section_4_2.py')

try:
    exec(open(path_4_2, encoding='utf-8').read())
except Exception as e:
    print(f"Error executing Python script {path_4_2}: {e}")
    sys.exit(1)

# ====================================================================
# STEP 4: Section 4.3 - Main Results
# ====================================================================
print("\n" + "="*50)
print("Step 5: Running Main Analysis of Section 4.3 (Python Analysis)")
print("="*50)

path_4_3 = os.path.join(ROOT_DIR, 'Code', 'Section4_3', '04_Main_Section_4_3.py')

try:
    exec(open(path_4_3, encoding='utf-8').read())
except Exception as e:
    print(f"Error executing Python script {path_4_3}: {e}")
    sys.exit(1)

# ====================================================================
# STEP 5: Section 4.2 - Robustness Check
# ====================================================================

print("\n" + "="*50)
print("Step 4: Running Robustness Checks of Section 4.2 (Python Analysis)")
print("="*50)

path_4_2_robust = os.path.join(ROOT_DIR, 'Code', 'Section4_2', '05_Appendix_G_4_2.py')

try:
    exec(open(path_4_2_robust, encoding='utf-8').read())
except Exception as e:
    print(f"Error executing Python script {path_4_2_robust}: {e}")
    sys.exit(1)

# ====================================================================
# STEP 6: Section 4.3 - Robustness Check
# ====================================================================
print("\n" + "="*50)
print("Step 6: Running Robustness Checks of Section 4.3 (Python Analysis)")
print("="*50)

path_4_3_robust = os.path.join(ROOT_DIR, 'Code', 'Section4_3', '06_Appendix_G_4_3.py')

try:
    exec(open(path_4_3_robust, encoding='utf-8').read())
except Exception as e:
    print(f"Error executing Python script {path_4_3_robust}: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("REPRODUCTION COMPLETED SUCCESSFULLY!")
print(f"Tables saved to: {output_path_table}")
print(f"Pictures saved to: {output_path_picture}")
print("="*50)
