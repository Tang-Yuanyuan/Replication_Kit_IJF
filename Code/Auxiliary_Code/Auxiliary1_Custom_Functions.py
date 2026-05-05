# Auxiliary3_Section_4.2_4.3.py
# ==============================================================
# Dependencies (self-contained; RANDOM_SEED injected at import
# time from run_all via the module's global namespace trick, OR
# defined here as a fallback default).
# ==============================================================

import os
import sys
import optuna
import pickle
import subprocess
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

# Allow run_all.py to override this by defining RANDOM_SEED before
# importing this module — the name will already exist in the calling
# module's namespace.  If this file is run standalone, fall back to 120.
try:
    RANDOM_SEED
except NameError:
    RANDOM_SEED = 120

# ------------------------------ Data Processing ------------------------------
def get_subset_data(data, mode="car"):
    """Extract the corresponding subset based on the mode."""
    if mode == "car":
        return data[data['publictrans'] < 5].copy()
    elif mode == "elec":
        return data[data['conditionernumber'] == 1].copy()
    elif mode == "green":
        return data[data['energy_consume2020'] > 1000].copy()
    else:
        raise ValueError("Invalid mode. Choose 'car', 'elec', or 'green'.")


# ------------------------------ Model Training ------------------------------
def train_xgb_model_bayesian(train_x, train_y, group_type="All", n_trials=50, cv=2):
    def objective(trial):
        if group_type == "All":
            # --- All ---
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=10),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
                "lambda": trial.suggest_float("reg_lambda", 0.8, 2.6),
                "gamma": trial.suggest_float("gamma", 1.3, 3.6),
            }
            
        else:
            # --- Demos ---
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=10),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
                "lambda": trial.suggest_float("reg_lambda", 0.8, 2.6),
                "gamma": trial.suggest_float("gamma", 1.3, 3.6),
            }

        param.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_SEED,
            "n_jobs": -1
        })
        
        model = xgb.XGBClassifier(**param)
        score = cross_val_score(model, train_x, train_y, cv=cv, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print(f"Performing Bayesian optimization for [{group_type}] group...")
    study.optimize(objective, n_trials=n_trials)
    
    final_params = study.best_params.copy()
    final_params.update({"objective": "binary:logistic", "eval_metric": "logloss", "random_state": RANDOM_SEED})
    
    best_model = xgb.XGBClassifier(**final_params)
    best_model.fit(train_x, train_y)
    
    return best_model


def train_xgb_regressor_bayesian(train_x, train_y, group_type="All", n_trials=50, cv=3):
    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 490, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
            "gamma": trial.suggest_float("gamma", 0, 2.0),
            "objective": "reg:squarederror", # 切换为回归损失函数
            "random_state": RANDOM_SEED,
            "n_jobs": -1
        }
        model = xgb.XGBRegressor(**param)
        score = cross_val_score(model, train_x, train_y, cv=cv, scoring='neg_mean_squared_error').mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_model = xgb.XGBRegressor(**study.best_params, random_state=RANDOM_SEED)
    best_model.fit(train_x, train_y)
    return best_model


# ------------------------------ Decision Table ------------------------------
def get_group_probabilities(model_dict, test_df, group_suffix="Demos"):
    prob_df = pd.DataFrame(index=test_df.index)
    for sub in ['Car', 'Elec', 'Green']:
        m_name = f"{sub}_{group_suffix}"
        if m_name in model_dict:
            model = model_dict[m_name]
            # 自动获取特征列名并预测
            feats = model.get_booster().feature_names
            prob_df[f'prob_{sub.lower()}'] = model.predict_proba(test_df[feats])[:, 1]
    return prob_df

def calculate_metrics(demos_df, all_df):
    """Compute performance comparison table across Demos and All groups."""
    cost_map = {7: 0, 6: 10, 5: 25, 4: 50, 3: 75, 2: 100}
    
    res = []
    for name, df in [("Demos Group", demos_df), ("All Group", all_df)]:
        valid_samples = df[df['is_valid'] == 1].copy()
        n_total = len(df)
        n_valid = len(valid_samples)

        accept_rate = valid_samples['is_accept'].mean() if n_valid > 0 else 0
        perfect_rate = valid_samples['is_perfect'].mean() if n_valid > 0 else 0

        accepted_samples = valid_samples[valid_samples['is_accept'] == 1]
        if len(accepted_samples) > 0:
            accept_cost = accepted_samples['real_wta_val'].map(cost_map).mean()
        else:
            accept_cost = 0
        
        res.append({
            "Group": name,
            "Total_Tests": n_total,
            "Valid_Predictions": n_valid,
            "Validity_Rate": n_valid / n_total,
            "Accept_Rate": accept_rate,
            "Perfect_Rate": perfect_rate,
            "Accept_Cost": accept_cost 
        })
    
    return pd.DataFrame(res).set_index("Group")


def get_ideal_results(test_df):
    """Simulate ideal scenario: respondents always select the option where y=1."""
    res = pd.DataFrame(index=test_df.index)
    
    res['best_option'] = 'None'
    for opt in ['car', 'elec', 'green']:
        res.loc[test_df[f'y_{opt}'] == 1, 'best_option'] = opt.capitalize()
    
    res['real_wta_val'] = np.nan
    for opt in ['car', 'elec', 'green']:
        mask = (res['best_option'] == opt.capitalize())
        res.loc[mask, 'real_wta_val'] = test_df.loc[mask, f'wta_{opt}']
    
    res['is_valid'] = (res['best_option'] != 'None').astype(int)
    res['is_accept'] = res['is_valid']
    res['is_perfect'] = res['is_valid']
    res['max_probability'] = 1.0  
    
    return res

def get_processed_results(probs_df, test_df, threshold=0.5):
    """
    Unified Processing Logic: Convert probabilities into decisions, and calculate `is_valid`, `is_accept`, and `is_perfect`.
    """
    res = probs_df.copy()

    res['max_probability'] = res[['prob_car', 'prob_elec', 'prob_green']].max(axis=1)
    res['best_option'] = res[['prob_car', 'prob_elec', 'prob_green']].idxmax(axis=1).str.replace('prob_', '')

    mask_car = (res['best_option'] == 'car') & (test_df['publictrans'] < 5)
    mask_elec = (res['best_option'] == 'elec') & (test_df['conditionernumber'] == 1)
    mask_green = (res['best_option'] == 'green') & (test_df['energy_consume2020'] > 1000)
    
    res['is_valid'] = ((mask_car | mask_elec | mask_green) & (res['max_probability'] > threshold)).astype(int)

    res['real_wta_val'] = np.nan
    for opt in ['car', 'elec', 'green']:
        mask = (res['best_option'] == opt)
        res.loc[mask, 'real_wta_val'] = test_df.loc[mask, f'wta_{opt}']

    res['is_accept'] = ((res['is_valid'] == 1) & (res['real_wta_val'] > 1)).astype(int)

    res['is_perfect'] = 0
    for opt in ['car', 'elec', 'green']:
        mask = (res['best_option'] == opt) & (res['is_valid'] == 1)
        res.loc[mask, 'is_perfect'] = test_df.loc[mask, f'y_{opt}']
        
    return res


def run_monte_carlo_random(test_df, n_iterations=200):
    """
    Perform *n* random simulations and calculate the average of the various metrics.
    """
    all_metrics = []
    
    cost_map = {7: 0, 6: 10, 5: 25, 4: 50, 3: 75, 2: 100}
    
    can_car = (test_df['publictrans'] < 5)
    can_elec = (test_df['conditionernumber'] == 1)
    can_green = (test_df['energy_consume2020'] > 1000)

    for i in range(n_iterations):
        np.random.seed(RANDOM_SEED + i)
        
        res = pd.DataFrame(index=test_df.index)
        res['best_option'] = 'None'
        
        for idx in test_df.index:
            available = []
            if can_car.loc[idx]: available.append('car')
            if can_elec.loc[idx]: available.append('elec')
            if can_green.loc[idx]: available.append('green')
            
            if available:
                res.loc[idx, 'best_option'] = np.random.choice(available)
        
        res['is_valid'] = (res['best_option'] != 'None').astype(int)
        valid_df = res[res['is_valid'] == 1].copy()
        
        if len(valid_df) > 0:
            res['real_wta_val'] = np.nan
            res['is_perfect'] = 0
            for opt in ['car', 'elec', 'green']:
                mask = (res['best_option'] == opt)
                if mask.any():
                    res.loc[mask, 'real_wta_val'] = test_df.loc[mask, f'wta_{opt}']
                    res.loc[mask, 'is_perfect'] = (test_df.loc[mask, f'y_{opt}'] == 1).astype(int)
            
            accept_mask = (res['is_valid'] == 1) & (res['real_wta_val'] > 1)
            is_accept_all = accept_mask.astype(int)
            
            n_valid = len(valid_df)
            round_accept_rate = is_accept_all.sum() / n_valid
            round_perfect_rate = (res['is_perfect']).sum() / n_valid
            
            accepted_wta = res.loc[accept_mask == 1, 'real_wta_val']
            round_cost = accepted_wta.map(cost_map).mean() if len(accepted_wta) > 0 else 0
            
            all_metrics.append({
                "Validity_Rate": n_valid / len(test_df),
                "Accept_Rate": round_accept_rate,
                "Perfect_Rate": round_perfect_rate,
                "Accept_Cost": round_cost
            })

    avg_results = pd.DataFrame(all_metrics).mean()
    
    final_random_df = pd.DataFrame([ {
        "Group": "Random (Lower Bound)",
        "Total_Tests": len(test_df),
        "Valid_Predictions": int(avg_results['Validity_Rate'] * len(test_df)),
        "Validity_Rate": avg_results['Validity_Rate'],
        "Accept_Rate": avg_results['Accept_Rate'],
        "Perfect_Rate": avg_results['Perfect_Rate'],
        "Accept_Cost": avg_results['Accept_Cost']
    } ]).set_index("Group")
    
    return final_random_df

format_dict = {
    'Validity_Rate': '{:.2%}',
    'Accept_Rate': '{:.2%}',
    'Perfect_Rate': '{:.2%}',
    'Accept_Cost': '{:.2f} 元'
}

def append_final_wta_column(results_df, test_df, xgb_reg_dict, group_suffix="All"):
    """
    Based on the `best_option` from the classification model, 
    extract the corresponding predicted WTA value from the regression model and round it to the nearest integer.
    """
    df = results_df.copy()

    temp_cols = []
    for opt in ['car', 'elec', 'green']:
        m_key = f"{opt.capitalize()}_{group_suffix}_Reg"
        col_name = f'temp_pred_{opt}'
        
        if m_key in xgb_reg_dict:
            model = xgb_reg_dict[m_key]
            feats = model.get_booster().feature_names
            df[col_name] = np.floor(model.predict(test_df[feats])).astype(int)
            temp_cols.append(col_name)
    
    df['pred_wta_val'] = np.nan
    
    for opt in ['car', 'elec', 'green']:
        mask = (df['best_option'].str.lower() == opt)
        temp_col = f'temp_pred_{opt}'
        if temp_col in df.columns:
            df.loc[mask, 'pred_wta_val'] = df.loc[mask, temp_col]

    df.drop(columns=temp_cols, inplace=True)
    
    return df


def update_eco_results_with_floor(results_df, preds_path):
    """
    results_df: The previously generated table containing `best_option` and `is_valid`.
    preds_path: The path to the R-exported `wta_preds_all.csv` or `demos.csv`.
    """
    preds_raw = pd.read_csv(preds_path)
    preds_raw.index = test_data.index

    opt_map = {
        'Car': 'pred_wta_car', 'car': 'pred_wta_car',
        'Elec': 'pred_wta_elec', 'elec': 'pred_wta_elec',
        'Green': 'pred_wta_green', 'green': 'pred_wta_green'
    }

    for idx in results_df.index:
        best_opt = results_df.loc[idx, 'best_option']
        
        if results_df.loc[idx, 'is_valid'] == 1 and pd.notna(best_opt):
            col_name = opt_map.get(best_opt)
            
            if col_name is not None:
                raw_val = preds_raw.loc[idx, col_name]
                results_df.loc[idx, 'pred_wta_val'] = np.floor(raw_val)
            else:
                results_df.loc[idx, 'pred_wta_val'] = np.nan
        else:
            results_df.loc[idx, 'pred_wta_val'] = np.nan
            
    return results_df


# ------------------------------ Empirical 4.2.2 ------------------------------
def get_n_household_metrics(results_df, test_df, n_list=[50, 100, 150], sort=True):
    cost_map = {7: 0, 6: 10, 5: 25, 4: 50, 3: 75, 2: 100, 1: 0}
    eval_pool = results_df[results_df['is_valid'] == 1].copy()
    
    eval_pool['real_wta_code'] = np.nan
    for opt in ['car', 'elec', 'green']:
        mask = (eval_pool['best_option'].str.lower() == opt)
        if mask.any():
            target_indices = eval_pool[mask].index
            eval_pool.loc[mask, 'real_wta_code'] = test_df.loc[target_indices, f'wta_{opt}']
    
    if sort:
        sort_cols = []
        if 'pred_wta_val' in eval_pool.columns: sort_cols.append('pred_wta_val')
        elif 'real_wta_val' in eval_pool.columns: sort_cols.append('real_wta_val')
        if 'max_probability' in eval_pool.columns: sort_cols.append('max_probability')
        if sort_cols:
            eval_pool = eval_pool.sort_values(by=sort_cols, ascending=False)
            
    summary_rows = []
    for n in n_list:
        if len(eval_pool) >= n:
            winners = eval_pool.head(n)
            valid_winners = winners.dropna(subset=['real_wta_code'])
            accepted_winners = valid_winners[valid_winners['real_wta_code'] > 1]
            accept_count = len(accepted_winners)
            
            summary_rows.append({
                "Quota (N)": n,
                "Accept_Rate": accept_count / n,
                "Accept_Cost": accepted_winners['real_wta_code'].map(cost_map).mean() if accept_count > 0 else 0.0
            })
        else:
            summary_rows.append({"Quota (N)": n, "Accept_Rate": np.nan, "Accept_Cost": np.nan})
    return pd.DataFrame(summary_rows)


def get_random_baseline_monte_carlo(test_df, n_list, iterations=1000):
    print(f"Performing {iterations} Monte Carlo random simulations...")
    all_sim_results = []
    
    for i in range(iterations):
        random_results = []
        for idx, row in test_df.iterrows():
            eligible = []
            if row['publictrans'] < 5: eligible.append('Car')
            if row['conditionernumber'] == 1: eligible.append('Elec')
            if row['energy_consume2020'] > 1000: eligible.append('Green')
            
            if not eligible:
                random_results.append({'best_option': 'None', 'is_valid': 0})
            else:
                random_results.append({'best_option': np.random.choice(eligible), 'is_valid': 1})
        
        df_sim = pd.DataFrame(random_results, index=test_df.index)
        
        metrics = get_n_household_metrics(df_sim, test_df, n_list=n_list, sort=False)
        all_sim_results.append(metrics)
    
    combined_sim = pd.concat(all_sim_results)
    mean_random_metrics = combined_sim.groupby("Quota (N)").mean().reset_index()
    return mean_random_metrics


# ------------------------------ Empirical 4.2.3 ------------------------------
def get_budget_metrics(results_df, test_df, budget_list=[1000, 2000, 3000, 4000], sort=True):
    cost_map = {7: 0, 6: 10, 5: 25, 4: 50, 3: 75, 2: 100, 1: 0}
    eval_pool = results_df[results_df['is_valid'] == 1].copy()
    eval_pool['real_wta_code'] = np.nan
    for opt in ['car', 'elec', 'green']:
        mask = (eval_pool['best_option'].str.lower() == opt)
        if mask.any():
            target_indices = eval_pool[mask].index
            eval_pool.loc[mask, 'real_wta_code'] = test_df.loc[target_indices, f'wta_{opt}']
    
    eval_pool['individual_cost'] = eval_pool['real_wta_code'].map(cost_map)
    
    if sort:
        sort_cols = []
        if 'pred_wta_val' in eval_pool.columns: sort_cols.append('pred_wta_val')
        elif 'real_wta_val' in eval_pool.columns: sort_cols.append('real_wta_val')
        if 'max_probability' in eval_pool.columns: sort_cols.append('max_probability')
        if sort_cols:
            eval_pool = eval_pool.sort_values(by=sort_cols, ascending=False)
    
    budget_summary = []
    for limit in budget_list:
        eval_pool['cum_cost'] = eval_pool['individual_cost'].cumsum()
        winners = eval_pool[eval_pool['cum_cost'] <= limit]
        n_recruited = len(winners)
        if n_recruited > 0:
            accepted_winners = winners[winners['real_wta_code'] > 1]
            accept_rate = len(accepted_winners) / n_recruited
        else:
            accept_rate = 0.0
        budget_summary.append({
            "Budget_Limit": limit,
            "Total_Recruited (N)": n_recruited,
            "Accept_Rate": accept_rate
        })
    return pd.DataFrame(budget_summary)


def get_random_budget_baseline_monte_carlo(test_df, budget_list, iterations=1000):
    print(f"Performing {iterations} Monte Carlo stochastic simulations under budget constraints....")
    all_sim_results = []
    
    for i in range(iterations):
        random_results = []
        for idx, row in test_df.iterrows():
            eligible = []
            if row['publictrans'] < 5: eligible.append('Car')
            if row['conditionernumber'] == 1: eligible.append('Elec')
            if row['energy_consume2020'] > 1000: eligible.append('Green')
            
            if not eligible:
                random_results.append({'best_option': 'None', 'is_valid': 0})
            else:
                random_results.append({
                    'best_option': np.random.choice(eligible), 
                    'is_valid': 1
                })
        
        df_sim = pd.DataFrame(random_results, index=test_df.index)
        
        metrics = get_budget_metrics(df_sim, test_df, budget_list=budget_list, sort=False)
        all_sim_results.append(metrics)
    
    combined_sim = pd.concat(all_sim_results)
    mean_random_metrics = combined_sim.groupby("Budget_Limit").mean().reset_index()
    
    return mean_random_metrics


# ------------------------------ Empirical 4.3 ------------------------------
def simulate_policy_knowledge_upgrade(df):
    sim_df = df.copy()
    
    # ---------------------------------------------------------
    # Step 1: De-discretization
    # Convert integer levels 1–7 into floating-point intervals [level, level+1)
    # ---------------------------------------------------------
    wta_cols = ['wta_car', 'wta_elec', 'wta_green']
    for col in wta_cols:
        mask_not_max = (sim_df[col] < 7) & (sim_df[col] >= 1)
        sim_df.loc[mask_not_max, col] = sim_df.loc[mask_not_max, col] + np.random.uniform(0, 1, size=mask_not_max.sum())

    params = {
        'wta_car':   {'heard_not_know': (0.183, 0.186), 'heard_know': (0.697, 0.366)},
        'wta_elec':  {'heard_not_know': (0.641, 0.208), 'heard_know': (0.888, 0.363)},
        'wta_green': {'heard_not_know': (0.357, 0.159), 'heard_know': (0.972, 0.286)}
    }

    policy_col = 'know_about_carbon_policy'
    mask_never = (sim_df[policy_col] == 'never')
    mask_not_know = (sim_df[policy_col] == 'heard but do not know')
    
    # ---------------------------------------------------------
    # Step 2: Apply Causal Intervention
    # ---------------------------------------------------------
    for target_wta in wta_cols:
        m_k, se_k = params[target_wta]['heard_know']
        m_n, se_n = params[target_wta]['heard_not_know']
        
        # A. 'never' -> 'heard and know'
        if mask_never.any():
            delta_never = np.random.normal(m_k, se_k, size=mask_never.sum())
            sim_df.loc[mask_never, target_wta] += np.maximum(delta_never, 0)
            
        # B. 'heard but do not know' -> 'heard and know'
        if mask_not_know.any():
            size = mask_not_know.sum()
            rand_know = np.random.normal(m_k, se_k, size=size)
            rand_not_know = np.random.normal(m_n, se_n, size=size)
            delta_diff = rand_know - rand_not_know
            sim_df.loc[mask_not_know, target_wta] += np.maximum(delta_diff, 0)

        # ---------------------------------------------------------
        # Step 3: Re-discretization and Range Restriction
        # ---------------------------------------------------------
        sim_df[target_wta] = np.floor(sim_df[target_wta]).clip(lower=1, upper=7)

    # ---------------------------------------------------------
    # Step 4: Update Cognitive Labels and Decision Logic
    # ---------------------------------------------------------
    affected_mask = mask_never | mask_not_know
    sim_df.loc[affected_mask, policy_col] = 'heard and know'
    
    sim_df['max_wta'] = sim_df[wta_cols].max(axis=1)
    for wta_col, y_col in zip(wta_cols, ['y_car', 'y_elec', 'y_green']):
        sim_df[y_col] = ((sim_df[wta_col] == sim_df['max_wta']) & (sim_df['max_wta'] > 1)).astype(int)
    
    return sim_df


# ------------------------------ Drawing Utilities ------------------------------
def clean_pct(val):
    if isinstance(val, str):
        clean = val.replace('%', '').replace('CNY', '').replace(',', '').strip()
        return float(clean)
    return float(val)
    
def append_metrics(target_dict, df, row_idx):
        if 'accept_cost' not in target_dict:
            target_dict['accept_cost'] = []
            target_dict['accept_rate'] = []
        target_dict['accept_cost'].append(df.iloc[row_idx]['Accept_Cost'])
        target_dict['accept_rate'].append(df.iloc[row_idx]['Accept_Rate'])
