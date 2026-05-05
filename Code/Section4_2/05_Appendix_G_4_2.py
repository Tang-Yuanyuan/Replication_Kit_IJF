import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Auxiliary_Code'))

try:
    from Auxiliary1_Custom_Functions_Weighted import *
    print("Auxiliary functions imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# ====================================================================
# 2. Data Processing
# ====================================================================

df = pd.read_csv(weighted_data_path)
r_script_dir = os.path.join(ROOT_DIR, 'Code', 'Auxiliary_Code')
r_executable = R_EXE

# (1) Create new variable y
wta_cols = ['wta_car', 'wta_elec', 'wta_green']
y_cols = ['y_car', 'y_elec', 'y_green']
df['max_wta'] = df[wta_cols].max(axis=1)

for wta_col, y_col in zip(wta_cols, y_cols):
    df[y_col] = ((df[wta_col] == df['max_wta']) & (df['max_wta'] > 1)).astype(int)

# (2) Split training and test sets
df['stratify_key'] = df['y_car'].astype(str) + df['y_elec'].astype(str) + df['y_green'].astype(str)

train_data, test_data = train_test_split(
    df,
    test_size=0.3,
    random_state=RANDOM_SEED,
    stratify=df['stratify_key']
)

train_data = train_data.drop(columns=['stratify_key'])
test_data  = test_data.drop(columns=['stratify_key'])

print(f"Proportion in original data | y_car: {df['y_car'].mean():.4f} | y_elec: {df['y_elec'].mean():.4f} | y_green: {df['y_green'].mean():.4f}")
print(f"Proportion in training set   | y_car: {train_data['y_car'].mean():.4f} | y_elec: {train_data['y_elec'].mean():.4f} | y_green: {train_data['y_green'].mean():.4f}")
print(f"Proportion in test set       | y_car: {test_data['y_car'].mean():.4f} | y_elec: {test_data['y_elec'].mean():.4f} | y_green: {test_data['y_green'].mean():.4f}")

train_data.to_csv(os.path.join(output_path_data, 'train_data.csv'), index=False, encoding='utf-8-sig')
test_data.to_csv(os.path.join(output_path_data, 'test_data.csv'),   index=False, encoding='utf-8-sig')
test_data_eco = test_data.copy()

# (3) XGBOOST data preparation: One-Hot Encode
categorical_cols = [
    'location', 'province', 'weekday',
    'heard_about_global_warming', 'know_about_low_carbon',
    'know_about_carbon_neutrality', 'know_about_carbon_policy'
]

train_data = pd.get_dummies(train_data, columns=categorical_cols, drop_first=False)
test_data  = pd.get_dummies(test_data,  columns=categorical_cols, drop_first=False)

train_data = train_data.astype(int, errors='ignore')
test_data  = test_data.astype(int, errors='ignore')

# (4) XGBOOST data preparation: Subset
train_car   = get_subset_data(train_data, mode="car")
train_elec  = get_subset_data(train_data, mode="elec")
train_green = get_subset_data(train_data, mode="green")

test_car    = get_subset_data(test_data, mode="car")
test_elec   = get_subset_data(test_data, mode="elec")
test_green  = get_subset_data(test_data, mode="green")

demos_all = [col for col in train_car.columns if col not in [
    "wta_car", "wta_elec", "wta_green", 'y_car', 'y_elec', 'y_green', "id",
    'weights', 'max_wta',
    'publictrans', 'conditionernumber', 'energy_consume2020',
]]

demos = [col for col in demos_all if not any(k_var in col for k_var in [
    'heard_about_global_warming',
    'know_about_low_carbon',
    'know_about_carbon_neutrality',
    'know_about_carbon_policy',
])]

exclude_car   = ['conditioner1month', 'mainuseelec']
exclude_elec  = ['caruse', 'mainuseelec']
exclude_green = ['caruse', 'conditioner1month']

demos_all_car   = [c for c in demos_all if c not in exclude_car]
demos_all_elec  = [c for c in demos_all if c not in exclude_elec]
demos_all_green = [c for c in demos_all if c not in exclude_green]

demos_car   = [c for c in demos if c not in exclude_car]
demos_elec  = [c for c in demos if c not in exclude_elec]
demos_green = [c for c in demos if c not in exclude_green]

# ====================================================================
# 3. Model Training (Y): Optuna Bayesian Optimization + Cross Validation
# ====================================================================
train_tasks = [
    {"name": "Car_Demos",   "train": train_car,   "y": "y_car",   "features": demos_car,      "type": "Demos"},
    {"name": "Car_All",     "train": train_car,   "y": "y_car",   "features": demos_all_car,  "type": "All"},
    {"name": "Elec_Demos",  "train": train_elec,  "y": "y_elec",  "features": demos_elec,     "type": "Demos"},
    {"name": "Elec_All",    "train": train_elec,  "y": "y_elec",  "features": demos_all_elec, "type": "All"},
    {"name": "Green_Demos", "train": train_green, "y": "y_green", "features": demos_green,    "type": "Demos"},
    {"name": "Green_All",   "train": train_green, "y": "y_green", "features": demos_all_green,"type": "All"},
]

trained_models = {}

for task in train_tasks:
    print(f"\nStarting Bayesian training: {task['name']}...")
    model = train_xgb_model_bayesian(
        task['train'][task['features']],
        task['train'][task['y']],
        group_type=task['type'],
        n_trials=50,
        cv=3
    )
    trained_models[task['name']] = model

print("\nAll models Bayesian training completed.")

params_list = []
for name, model in trained_models.items():
    p = model.get_params()
    params_list.append({
        "Model":         name,
        "n_estimators":  p.get('n_estimators'),
        "max_depth":     p.get('max_depth'),
        "learning_rate": f"{p.get('learning_rate'):.4f}",
        "reg_lambda":    f"{p.get('reg_lambda'):.4f}",
        "gamma":         f"{p.get('gamma'):.4f}",
    })

df_params = pd.DataFrame(params_list).set_index("Model")
df_params.to_csv(os.path.join(output_path_table, 'Table_C.1_PrefAlt.csv'), encoding='utf-8-sig')
print("Table_C.1_PrefAlt.csv saved.")

# ====================================================================
# 4. Data Frame Construction & Prediction Accuracy
# ====================================================================

threscut       = 0
set_iterations = 1000

# IDEAL
results_ideal = get_ideal_results(test_data)

raw_ideal_metrics        = calculate_metrics(results_ideal, results_ideal)
final_df_ideal           = raw_ideal_metrics.iloc[[0]].copy()
final_df_ideal.index     = ['Ideal (Upper Bound)']

# XGBOOST
demos_probs = get_group_probabilities(trained_models, test_data, "Demos")
all_probs   = get_group_probabilities(trained_models, test_data, "All")

results_xgb_demos = get_processed_results(demos_probs, test_data, threshold=threscut)
results_xgb_all   = get_processed_results(all_probs,   test_data, threshold=threscut)

final_df_xgb = calculate_metrics(results_xgb_demos, results_xgb_all)

# ECONOMETRIC
r_script_path = os.path.join(r_script_dir, 'Auxiliary2_Section_4.2_Weighted.R')

print("Automatically calling R engine to run script...")

try:
    process = subprocess.run(
        [r_executable, r_script_path],
        capture_output=True,
        text=True,
        check=True,
        encoding='utf-8',
        errors='ignore'
    )
    print("R executed successfully!")
except subprocess.CalledProcessError as e:
    print("R script execution error, please check the logic in R code:")
    print(e.stderr)
    raise

logit_demos = pd.read_csv(os.path.join(r_output_dir, 'logit_probs_demos.csv'))
logit_all   = pd.read_csv(os.path.join(r_output_dir, 'logit_probs_all.csv'))

logit_demos.index = test_data.index
logit_all.index   = test_data.index

results_eco_demos = get_processed_results(logit_demos, test_data, threshold=threscut)
results_eco_all   = get_processed_results(logit_all,   test_data, threshold=threscut)

final_df_eco = calculate_metrics(results_eco_demos, results_eco_all)

# Monte Carlo Random Baseline
final_df_random = run_monte_carlo_random(test_data, n_iterations=set_iterations)

accuracy_data = {
    'Logistic regression I':  clean_pct(final_df_eco.loc["Demos Group", 'Perfect_Rate']) * 100,
    'Logistic regression II': clean_pct(final_df_eco.loc["All Group",   'Perfect_Rate']) * 100,
    'XGBoost algorithm I':    clean_pct(final_df_xgb.loc["Demos Group", 'Perfect_Rate']) * 100,
    'XGBoost algorithm II':   clean_pct(final_df_xgb.loc["All Group",   'Perfect_Rate']) * 100,
}

df_acc = pd.Series(accuracy_data)

fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

x     = np.arange(len(df_acc))
width = 0.4

bars = ax.bar(x, df_acc.values, width, color='black')

ax.set_ylim(60, max(accuracy_data['XGBoost algorithm II'], accuracy_data['XGBoost algorithm I']) + 2)
ax.set_ylabel('Prediction Accuracy (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(df_acc.index, rotation=15, ha='right', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_path_picture, 'Figure_3.png'), bbox_inches='tight', dpi=300)
plt.close()
print("Figure_3.png saved.")

pd.DataFrame.from_dict(accuracy_data, orient='index', columns=['Perfect_Rate (%)'])\
  .to_csv(os.path.join(output_path_table, 'Table_D.2_PrefAlt.csv'), encoding='utf-8-sig')
print("Table_D.2_PrefAlt.csv saved.")

# ====================================================================
# 5. Empirical 4.2.1
# ====================================================================
ordered_keys = ['Random', 'Logit_All', 'Logit_Demos', 'XGB_All', 'XGB_Demos', 'Ideal']
labels = [
    'Random assignment',
    'Logistic regression II',
    'Logistic regression I',
    'XGBoost algorithm II',
    'XGBoost algorithm I',
    'Perfect assignment',
]

raw_data = {
    'Random': {
        'Rate': clean_pct(final_df_random['Accept_Rate'].iloc[0]) * 100,
        'Cost': clean_pct(final_df_random['Accept_Cost'].iloc[0]),
    },
    'Logit_Demos': {
        'Rate': clean_pct(final_df_eco.loc["Demos Group", 'Accept_Rate']) * 100,
        'Cost': clean_pct(final_df_eco.loc["Demos Group", 'Accept_Cost']),
    },
    'Logit_All': {
        'Rate': clean_pct(final_df_eco.loc["All Group", 'Accept_Rate']) * 100,
        'Cost': clean_pct(final_df_eco.loc["All Group", 'Accept_Cost']),
    },
    'XGB_Demos': {
        'Rate': clean_pct(final_df_xgb.loc["Demos Group", 'Accept_Rate']) * 100,
        'Cost': clean_pct(final_df_xgb.loc["Demos Group", 'Accept_Cost']),
    },
    'XGB_All': {
        'Rate': clean_pct(final_df_xgb.loc["All Group", 'Accept_Rate']) * 100,
        'Cost': clean_pct(final_df_xgb.loc["All Group", 'Accept_Cost']),
    },
    'Ideal': {
        'Rate': clean_pct(final_df_ideal['Accept_Rate'].iloc[0]) * 100,
        'Cost': clean_pct(final_df_ideal['Accept_Cost'].iloc[0]),
    },
}

df_plot = pd.DataFrame([raw_data[key] for key in ordered_keys]).apply(pd.to_numeric)

fig, ax1 = plt.subplots(figsize=(12, 7), dpi=120)
ax2 = ax1.twinx()

x        = np.arange(len(labels))
width    = 0.4
x_offset = 0.2

bars = ax1.bar(x, df_plot['Cost'], width, color='#d9d9d9', edgecolor='black', label='Average Compensation')

ax2.plot(x + x_offset, df_plot['Rate'], color='#7f7f7f', marker='o', linestyle='--',
         linewidth=2, markersize=8, label='Acceptance Rate', zorder=5)

ax1.set_ylim(df_plot['Cost'].min() - 2, df_plot['Cost'].max() + 2)
ax2.set_ylim(df_plot['Rate'].min() - 2, 100)

ax1.set_ylabel('Average Compensation (Yuan)', fontsize=12)
ax2.set_ylabel('Acceptance Rate (%)', fontsize=12)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=25, ha='right')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

for i, bar in enumerate(bars):
    ax1.annotate(f"{df_plot['Cost'].iloc[i]:.2f}",
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)

for i in range(len(df_plot)):
    current_y_offset = 12
    current_x_offset = x_offset
    if i == 4:
        current_y_offset = 15
        current_x_offset += 0.05
    ax2.annotate(f"{df_plot['Rate'].iloc[i]:.2f}%",
                 xy=(x[i] + current_x_offset, df_plot['Rate'].iloc[i]),
                 xytext=(0, current_y_offset), textcoords="offset points",
                 ha='center', fontsize=10, color='#444444', weight='bold')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(output_path_picture, 'Figure_4.png'), bbox_inches='tight', dpi=300)
plt.close()
print("Figure_4.png saved.")

pd.DataFrame(raw_data).T\
  .to_csv(os.path.join(output_path_table, 'Table_D.3.csv'), encoding='utf-8-sig')
print("Table_D.3.csv saved.")

# ====================================================================
# 6. Model Training (WTA): Optuna Bayesian Optimization + Cross Validation
# ====================================================================

trained_models_reg = {}

reg_tasks_reg = [
    {"name": "Car_Demos_Reg",   "train": train_car,   "y": "wta_car",   "features": demos_car,       "type": "Demos"},
    {"name": "Car_All_Reg",     "train": train_car,   "y": "wta_car",   "features": demos_all_car,   "type": "All"},
    {"name": "Elec_Demos_Reg",  "train": train_elec,  "y": "wta_elec",  "features": demos_elec,      "type": "Demos"},
    {"name": "Elec_All_Reg",    "train": train_elec,  "y": "wta_elec",  "features": demos_all_elec,  "type": "All"},
    {"name": "Green_Demos_Reg", "train": train_green, "y": "wta_green", "features": demos_green,     "type": "Demos"},
    {"name": "Green_All_Reg",   "train": train_green, "y": "wta_green", "features": demos_all_green, "type": "All"},
]

for task in reg_tasks_reg:
    print(f"\nStarting Bayesian regression training: {task['name']}...")
    model = train_xgb_regressor_bayesian(
        task['train'][task['features']],
        task['train'][task['y']],
        group_type=task['type'],
        n_trials=100,
        cv=3
    )
    trained_models_reg[task['name']] = model

    preds = model.predict(test_data[task['features']])
    mae   = np.mean(np.abs(test_data[task['y']] - preds))
    print(f"{task['name']} completed, test set MAE (Mean Absolute Error): {mae:.4f}")

print("\nAll regression models training completed.")

params_list_wta = []
for name, model in trained_models_reg.items():
    p_wta = model.get_params()
    params_list_wta.append({
        "Model":         name,
        "n_estimators":  p_wta.get('n_estimators'),
        "max_depth":     p_wta.get('max_depth'),
        "learning_rate": f"{p_wta.get('learning_rate'):.4f}",
        "reg_lambda":    f"{p_wta.get('reg_lambda'):.4f}",
        "gamma":         f"{p_wta.get('gamma'):.4f}",
    })

df_params_wta = pd.DataFrame(params_list_wta).set_index("Model")
df_params_wta.to_csv(os.path.join(output_path_table, 'Table_C.1_WTA.csv'), encoding='utf-8-sig')
print("Table_C.1_WTA.csv saved.")

# ====================================================================
# 7. Update decision tables (with predicted wta)
# ====================================================================

# XGB
results_xgb_all   = append_final_wta_column(results_xgb_all,   test_data, trained_models_reg, "All")
results_xgb_demos = append_final_wta_column(results_xgb_demos, test_data, trained_models_reg, "Demos")
print("Extracted corresponding predicted WTA based on decision (floor rounding).")
print("Machine learning model (XGB) results processed, corresponding df updated.")

# ECO
r_script_path_wta = os.path.join(r_script_dir, 'Auxiliary3_Section_4.2_Weighted.R')

print("Running R economic model (Ordered Logit)...")
try:
    result = subprocess.run(
        [r_executable, r_script_path_wta],
        capture_output=True,
        text=True,
        check=True,
        encoding='utf-8',
        errors='ignore'
    )
    print("R script executed successfully!")
except subprocess.CalledProcessError as e:
    print(f"R script execution failed! Error message:\n{e.stderr}")
except Exception as e:
    print(f"Other error occurred: {e}")

results_eco_all   = update_eco_results_with_floor(results_eco_all,   os.path.join(r_output_dir, 'wta_preds_all.csv'))
results_eco_demos = update_eco_results_with_floor(results_eco_demos, os.path.join(r_output_dir, 'wta_preds_demos.csv'))
print("Extracted corresponding predicted WTA based on decision (floor rounding).")
print("Economic model (Eco) results processed, corresponding df updated.")

# ====================================================================
# 8. Empirical 4.2.2 Results for specified household numbers
# ====================================================================
current_n_list = [55, 100, 145]

m_ideal     = get_n_household_metrics(results_ideal,     test_data, n_list=current_n_list)
m_xgb_all   = get_n_household_metrics(results_xgb_all,   test_data, n_list=current_n_list)
m_xgb_demos = get_n_household_metrics(results_xgb_demos, test_data, n_list=current_n_list)
m_eco_all   = get_n_household_metrics(results_eco_all,   test_data, n_list=current_n_list)
m_eco_demos = get_n_household_metrics(results_eco_demos, test_data, n_list=current_n_list)
m_random    = get_random_baseline_monte_carlo(test_data, n_list=current_n_list, iterations=set_iterations)

m_ideal['Group']     = '1. Ideal (Upper Bound)'
m_xgb_all['Group']   = '2.1 XGBoost (All)'
m_xgb_demos['Group'] = '2.2 XGBoost (Demos)'
m_eco_all['Group']   = '3.1 Ordered Logit (All)'
m_eco_demos['Group'] = '3.2 Ordered Logit (Demos)'
m_random['Group']    = '4. Random (1000x Mean)'

all_metrics = pd.concat([m_ideal, m_xgb_all, m_xgb_demos, m_eco_all, m_eco_demos, m_random])

final_pivot = all_metrics.pivot(index='Group', columns='Quota (N)')

auto_format_dict = {}
for col in final_pivot.columns:
    metric_name, n_val = col
    if "Accept_Rate" in metric_name:
        auto_format_dict[col] = "{:.2%}"
    elif "Accept_Cost" in metric_name:
        auto_format_dict[col] = "{:.2f} Yuan"

print(f"\nPolicy simulation comparison results (Quota N = {current_n_list})")
styled_table = final_pivot.style.format(auto_format_dict)

cost_data = final_pivot.xs('Accept_Cost', axis=1)
quota_n   = cost_data.columns.tolist()

fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(10, 8),
    gridspec_kw={'height_ratios': [3, 1]}, dpi=120
)
fig.subplots_adjust(hspace=0.1)

styles = {
    '4. Random (1000x Mean)':    {'color': 'gray',  'linestyle': ':',  'marker': '',  'label': 'Random assignment'},
    '3.2 Ordered Logit (Demos)': {'color': 'gray',  'linestyle': '-',  'marker': 'o', 'label': 'Logistic regression I'},
    '2.2 XGBoost (Demos)':       {'color': 'black', 'linestyle': '-',  'marker': 'o', 'label': 'XGBoost algorithm I'},
    '3.1 Ordered Logit (All)':   {'color': 'gray',  'linestyle': '-.', 'marker': '^', 'label': 'Logistic regression II'},
    '2.1 XGBoost (All)':         {'color': 'black', 'linestyle': '-.', 'marker': '^', 'label': 'XGBoost algorithm II'},
    '1. Ideal (Upper Bound)':    {'color': 'black', 'linestyle': '--', 'marker': '',  'label': 'Perfect assignment'},
}

for group_name, style in styles.items():
    if group_name in cost_data.index:
        y_values = cost_data.loc[group_name].values
        ax_top.plot(quota_n, y_values, **style, linewidth=1.5, markersize=7)
        ax_bottom.plot(quota_n, y_values, **style, linewidth=1.5, markersize=7)

ax_top.set_ylim(
    min(m_xgb_demos.loc[0]['Accept_Cost'], m_xgb_all.loc[0]['Accept_Cost']) - 1.5,
    m_random.loc[0]['Accept_Cost'] + 2
)
ax_bottom.set_ylim(0, m_ideal.loc[0]['Accept_Cost'] + 5)

ax_top.spines['bottom'].set_visible(False)
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_bottom.spines['right'].set_visible(False)

ax_top.tick_params(labeltop=False)
ax_bottom.xaxis.tick_bottom()

d = .015
kwargs = dict(transform=ax_top.transAxes, color='black', clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)

fig.text(0.04, 0.5, 'Compensation Spending (Yuan/month/household)',
         va='center', rotation='vertical', fontsize=12)
ax_bottom.set_xlabel('Targeted Numbers of Households', fontsize=12)
ax_bottom.set_xticks(quota_n)

ax_top.legend(loc='center left', bbox_to_anchor=(1.05, 0.3), frameon=False, fontsize=11)

plt.savefig(os.path.join(output_path_picture, 'Figure_5.png'), bbox_inches='tight', dpi=300)
plt.close()
print("Figure_5.png saved.")

final_pivot.to_csv(os.path.join(output_path_table, 'Table_D.4.csv'), encoding='utf-8-sig')
print("Table_D.4.csv saved.")

# ====================================================================
# 9. Empirical 4.2.3 Results for limited budget
# ====================================================================

current_budgets = [1000, 2000, 3000, 4000]

bm_ideal     = get_budget_metrics(results_ideal,     test_data, budget_list=current_budgets)
bm_xgb_all   = get_budget_metrics(results_xgb_all,   test_data, budget_list=current_budgets)
bm_xgb_demos = get_budget_metrics(results_xgb_demos, test_data, budget_list=current_budgets)
bm_eco_all   = get_budget_metrics(results_eco_all,   test_data, budget_list=current_budgets)
bm_eco_demos = get_budget_metrics(results_eco_demos, test_data, budget_list=current_budgets)
bm_random    = get_random_budget_baseline_monte_carlo(test_data, current_budgets, iterations=set_iterations)

bm_ideal['Group']     = '1. Ideal (Theoretical Max)'
bm_xgb_all['Group']   = '2.1 XGBoost (All Features)'
bm_xgb_demos['Group'] = '2.2 XGBoost (Demos Only)'
bm_eco_all['Group']   = '3.1 Ordered Logit (All Features)'
bm_eco_demos['Group'] = '3.2 Ordered Logit (Demos Only)'
bm_random['Group']    = '4. Random (1000x Mean)'

budget_final = pd.concat([bm_ideal, bm_xgb_all, bm_xgb_demos, bm_eco_all, bm_eco_demos, bm_random])

budget_pivot = budget_final.pivot(index='Group', columns='Budget_Limit')
budget_pivot[('Accept_Rate', 'Average')] = budget_pivot['Accept_Rate'].mean(axis=1)

b_format_dict = {}
for col in budget_pivot.columns:
    if "Accept_Rate" in col[0]:
        b_format_dict[col] = "{:.2%}"
    else:
        b_format_dict[col] = "{:.1f} households"

plot_df         = budget_pivot['Total_Recruited (N)']
current_budgets = plot_df.columns.tolist()
avg_rates       = budget_pivot[('Accept_Rate', 'Average')]

fig, ax = plt.subplots(figsize=(12, 7), dpi=120)

styles = {
    '1. Ideal (Theoretical Max)':       {'color': 'black', 'linestyle': '--', 'marker': '',  'label_base': 'Perfect assignment'},
    '2.2 XGBoost (Demos Only)':         {'color': 'black', 'linestyle': '-.',  'marker': '^', 'label_base': 'XGBoost algorithm I'},
    '2.1 XGBoost (All Features)':       {'color': 'black', 'linestyle': '-',  'marker': 'o', 'label_base': 'XGBoost algorithm II'},
    '3.2 Ordered Logit (Demos Only)':   {'color': 'gray',  'linestyle': '-.',  'marker': '^', 'label_base': 'Logistic regression I'},
    '3.1 Ordered Logit (All Features)': {'color': 'gray',  'linestyle': '-',  'marker': 'o', 'label_base': 'Logistic regression II'},
    '4. Random (1000x Mean)':           {'color': 'gray',  'linestyle': ':',  'marker': '',  'label_base': 'Random assignment'},
}

for group_name, style in styles.items():
    if group_name in plot_df.index:
        y_values  = plot_df.loc[group_name].values
        rate_val  = avg_rates.loc[group_name]
        full_label = (
            f"{style['label_base']} ({rate_val:.2%})"
            if isinstance(rate_val, float)
            else f"{style['label_base']} ({rate_val})"
        )
        ax.plot(current_budgets, y_values,
                color=style['color'], linestyle=style['linestyle'],
                marker=style['marker'], label=full_label,
                linewidth=1.8, markersize=8)

ax.set_xlabel('Mitigation Budget (Yuan/month)', fontsize=12)
ax.set_ylabel('Number of Participating Households', fontsize=12)
ax.set_xticks(current_budgets)
ax.set_xlim(min(current_budgets) - 200, max(current_budgets) + 200)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(plot_df.min().min() - 5, plot_df.max().max() + 5)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_path_picture, 'Figure_6.png'), bbox_inches='tight', dpi=300)
plt.close()
print("Figure_6.png saved.")

budget_pivot.to_csv(os.path.join(output_path_table, 'Table_D.5.csv'), encoding='utf-8-sig')
print("Table_D.5.csv saved.")