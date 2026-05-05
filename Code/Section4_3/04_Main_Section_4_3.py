threscut      = 0
n_iterations  = 200
all_iteration_results = []
required_cols = [
    'know_about_carbon_policy_never',
    'know_about_carbon_policy_heard but do not know',
    'know_about_carbon_neutrality_heard but do not know',
    'know_about_carbon_neutrality_never',
]

# -- paths (all resolved relative to ROOT_DIR set in run_all.py) --------------
r_script_path_sm   = os.path.join(ROOT_DIR, 'Code', 'Auxiliary_Code', 'Auxiliary4_Section_4.3.R')
r_output_dir_43    = os.path.join(ROOT_DIR, 'Data', 'Intermediate')  
sim_csv_path       = os.path.join(r_output_dir_43, 'test_data_simulated.csv')

test_base = test_data_eco.copy().reset_index(drop=True)

print(f"Starting {n_iterations} knowledge growth simulations...")

simulate4_3_results = {
    'ideal_sm':     {},
    'random_sm':    {},
    'xgb_all_sm':   {},
    'xgb_demos_sm': {},
    'eco_all_sm':   {},
    'eco_demos_sm': {},
}

check_results = {'wta_car': [], 'wta_elec': [], 'wta_green': []}

for i in tqdm(range(n_iterations)):
    np.random.seed(RANDOM_SEED + i)

    # 1 Update Test_data (WTA + knowledge improvement)
    simulated_tb = simulate_policy_knowledge_upgrade(test_base.copy())
    check_results['wta_car'].append(simulated_tb['wta_car'].mean())
    check_results['wta_elec'].append(simulated_tb['wta_elec'].mean())
    check_results['wta_green'].append(simulated_tb['wta_green'].mean())

    # 2 ECO part -- invoke R script
    os.makedirs(os.path.dirname(sim_csv_path), exist_ok=True)
    simulated_tb.to_csv(sim_csv_path, index=False, encoding='utf-8-sig')

    try:
        subprocess.run(
            [r_executable, r_script_path_sm],
            capture_output=True, text=True, check=True,
            encoding='utf-8', errors='ignore',
        )
    except subprocess.CalledProcessError as e:
        print("R script execution error:")
        print(e.stderr)
        raise

    logit_demos_sm = pd.read_csv(os.path.join(r_output_dir_43, 'logit_probs_demos_simulated.csv'))
    logit_all_sm   = pd.read_csv(os.path.join(r_output_dir_43, 'logit_probs_all_simulated.csv'))
    logit_demos_sm.index = simulated_tb.index
    logit_all_sm.index   = simulated_tb.index

    results_eco_demos_sm = get_processed_results(logit_demos_sm, simulated_tb, threshold=threscut)
    results_eco_all_sm   = get_processed_results(logit_all_sm,   simulated_tb, threshold=threscut)
    final_df_eco_sm      = calculate_metrics(results_eco_demos_sm, results_eco_all_sm)

    # 3 Prepare dummies for non-R models
    simulated_tb = pd.get_dummies(simulated_tb, columns=categorical_cols, drop_first=False)
    simulated_tb = simulated_tb.astype(int, errors='ignore')
    for col in required_cols:
        if col not in simulated_tb.columns:
            simulated_tb[col] = 0

    # 4 Ideal
    results_ideal_sm     = get_ideal_results(simulated_tb)
    raw_ideal_metrics_sm = calculate_metrics(results_ideal_sm, results_ideal_sm)
    final_df_ideal_sm    = raw_ideal_metrics_sm.iloc[[0]].copy()
    final_df_ideal_sm.index = ['Ideal (Upper Bound)']

    # 5 Random
    final_df_random_sm = run_monte_carlo_random(simulated_tb, n_iterations=set_iterations)

    # 6 XGBoost
    demos_probs_sm        = get_group_probabilities(trained_models, simulated_tb, "Demos")
    all_probs_sm          = get_group_probabilities(trained_models, simulated_tb, "All")
    results_xgb_demos_sm  = get_processed_results(demos_probs_sm, simulated_tb, threshold=threscut)
    results_xgb_all_sm    = get_processed_results(all_probs_sm,   simulated_tb, threshold=threscut)
    final_df_xgb_sm       = calculate_metrics(results_xgb_demos_sm, results_xgb_all_sm)

    # 7 Summary
    append_metrics(simulate4_3_results['ideal_sm'],     final_df_ideal_sm,  0)
    append_metrics(simulate4_3_results['random_sm'],    final_df_random_sm, 0)
    append_metrics(simulate4_3_results['xgb_demos_sm'], final_df_xgb_sm,    0)
    append_metrics(simulate4_3_results['xgb_all_sm'],   final_df_xgb_sm,    1)
    append_metrics(simulate4_3_results['eco_demos_sm'], final_df_eco_sm,    0)
    append_metrics(simulate4_3_results['eco_all_sm'],   final_df_eco_sm,    1)

# ====================================================================
# Post-simulation: build summary & Figure 7
# ====================================================================

name_map = {
    'ideal_sm':     'Perfect assignment',
    'random_sm':    'Random assignment',
    'xgb_all_sm':   'XGBoost algorithm II',
    'xgb_demos_sm': 'XGBoost algorithm I',
    'eco_all_sm':   'Logistic regression II',
    'eco_demos_sm': 'Logistic regression I',
}

summary_stats = []
for scenario, data in simulate4_3_results.items():
    costs = data.get('accept_cost', [])
    rates = data.get('accept_rate', [])
    if costs and rates:
        summary_stats.append({
            'Scenario': name_map[scenario],
            'Cost_Mean': np.mean(costs),
            'Cost_Std':  np.std(costs),
            'Rate_Mean': np.mean(rates),
            'Rate_Std':  np.std(rates),
            'Cost_CV':   np.std(costs) / np.mean(costs) if np.mean(costs) != 0 else 0,
        })
df_summary = pd.DataFrame(summary_stats)

# Baseline (single-run results from Section 4.2)
simulate4_3_results_baseline = {
    'ideal':     {}, 'random':    {},
    'xgb_all':   {}, 'xgb_demos': {},
    'eco_all':   {}, 'eco_demos': {},
}
append_metrics(simulate4_3_results_baseline['ideal'],     final_df_ideal,  0)
append_metrics(simulate4_3_results_baseline['random'],    final_df_random, 0)
append_metrics(simulate4_3_results_baseline['xgb_demos'], final_df_xgb,    0)
append_metrics(simulate4_3_results_baseline['xgb_all'],   final_df_xgb,    1)
append_metrics(simulate4_3_results_baseline['eco_demos'], final_df_eco,    0)
append_metrics(simulate4_3_results_baseline['eco_all'],   final_df_eco,    1)

name_map_baseline = {
    'ideal':     'Perfect assignment',
    'random':    'Random assignment',
    'xgb_all':   'XGBoost algorithm II',
    'xgb_demos': 'XGBoost algorithm I',
    'eco_all':   'Logistic regression II',
    'eco_demos': 'Logistic regression I',
}
baseline_list = [
    {'Scenario': name_map_baseline[k], 'B_Cost': v['accept_cost'][0], 'B_Rate': v['accept_rate'][0]}
    for k, v in simulate4_3_results_baseline.items()
]
df_baseline = pd.DataFrame(baseline_list)

ordered_scenarios = [
    'Random assignment',
    'Logistic regression II',
    'Logistic regression I',
    'XGBoost algorithm II',
    'XGBoost algorithm I',
    'Perfect assignment',
]

plot_df = pd.merge(df_summary, df_baseline, on='Scenario')
plot_df['Scenario'] = pd.Categorical(plot_df['Scenario'], categories=ordered_scenarios, ordered=True)
plot_df = plot_df.sort_values('Scenario').reset_index(drop=True)

n_sim  = len(simulate4_3_results['ideal_sm']['accept_cost'])
t_val  = stats.t.ppf(0.975, n_sim - 1)
plot_df['Cost_CI']      = t_val * plot_df['Cost_Std']
plot_df['Rate_CI_Pct']  = t_val * plot_df['Rate_Std'] * 100
plot_df['Rate_Mean_Pct'] = plot_df['Rate_Mean'] * 100
plot_df['B_Rate_Pct']    = plot_df['B_Rate']    * 100

fig, ax1 = plt.subplots(figsize=(15, 8), dpi=120)
ax2 = ax1.twinx()

x     = np.arange(len(plot_df))
width = 0.32

bar1 = ax1.bar(x - width/2, plot_df['B_Cost'], width,
               label='Benchmark Average Compensation',
               color='#E0E0E0', edgecolor='black', linewidth=0.8, zorder=2)
bar2 = ax1.bar(x + width/2, plot_df['Cost_Mean'], width,
               label='Counterfactual Average Compensation',
               color='#5E5E5E', edgecolor='black', linewidth=0.8,
               yerr=plot_df['Cost_CI'], capsize=4,
               error_kw={'elinewidth': 1.2}, zorder=2)

ax2.plot(x, plot_df['B_Rate_Pct'],    color='#808080', linestyle='--', marker='o',
         markersize=7, label='Benchmark Acceptance Rate',       alpha=0.9, zorder=3)
ax2.plot(x, plot_df['Rate_Mean_Pct'], color='black',   linestyle='-',  marker='s',
         markersize=7, linewidth=1.5, label='Counterfactual Acceptance Rate', zorder=4)
ax2.fill_between(x,
                 plot_df['Rate_Mean_Pct'] - plot_df['Rate_CI_Pct'],
                 plot_df['Rate_Mean_Pct'] + plot_df['Rate_CI_Pct'],
                 color='black', alpha=0.1, label='95% Confidence Interval')

ax1.set_ylabel('Average Compensation (Yuan)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Acceptance Rate (%)',       fontsize=12, fontweight='bold')
ax1.set_ylim(df_summary['Cost_Mean'].min() - 4.3, df_baseline['B_Cost'].max() + 3)
ax2.set_ylim(82, 100)
ax1.set_xticks(x)
ax1.set_xticklabels(plot_df['Scenario'], rotation=15, ha='right', fontsize=10)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9)

autolabel(bar1)
autolabel(bar2)

for i, val in enumerate(plot_df['Rate_Mean_Pct']):
    ax2.annotate(f'{val:.2f}%', (x[i], val),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', fontsize=9, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=3, frameon=False)

plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
plt.tight_layout()

# -- Save Figure 7 ------------------------------------------------------------
figure7_path = os.path.join(output_path_picture, 'Figure_7.png')
plt.savefig(figure7_path, bbox_inches='tight', dpi=300)
plt.show()
print(f"Figure 7 saved -> {figure7_path}")
