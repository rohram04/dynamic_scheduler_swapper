import pandas as pd

# Load your CSV
csv_file = "./bare_metal/summary.csv"  # replace with your path
num_cores = 4  # set this to your bare metal core count

df = pd.read_csv(csv_file, index_col=0)

# Columns of interest
columns_of_interest = [
    "T_cpu_percent_avg", "T_iowait_percent_avg", "T_mem_used_avg", "T_mem_available_avg",
    "T_mem_available_delta", "T_mem_available_pct_change", "T_swap_used_avg", "T_cache_mem_avg",
    "T_cache_mem_delta", "T_cache_mem_pct_change", "T_buffers_mem_avg", "T_buffers_mem_delta",
    "T_buffers_mem_pct_change", "T_io_write_total_delta", "T_io_write_total_pct_change",
    "T_nvcsw_total_delta", "T_nvcsw_total_pct_change", "T_vcsw_total_delta", "T_vcsw_total_pct_change",
    "Average_TAT", "Average_RT", "CV_Fairness"
]

# Keep only columns of interest
df_selected = df[columns_of_interest]

# Keep only 'mean' and 'std' rows
df_stats = df_selected.loc[["mean", "std"]]

# === Split metrics into CPU/RT/context-switch metrics and the rest ===
cpu_related_cols = [
    "Average_RT",
    "T_nvcsw_total_delta", "T_nvcsw_total_pct_change",
    "T_vcsw_total_delta", "T_vcsw_total_pct_change"
]

other_cols = [col for col in df_selected.columns if col not in cpu_related_cols]

# --- CPU-related metrics, normalized per core ---
df_cpu = df_stats[cpu_related_cols].T

# Add normalized mean
df_cpu["normalized_mean"] = df_cpu["mean"] / num_cores
df_cpu["cov_%"] = round((df_cpu["std"] / df_cpu["normalized_mean"]) * 100, 2)
df_cpu = df_cpu[["mean", "normalized_mean", "std", "cov_%"]]
# df_cpu = df_cpu[["mean", "std", "cov_%"]]

# Save and print
df_cpu.to_csv("./bare_metal/cpu_variability.csv")
print("### CPU / RT / Context-Switch Metrics (normalized per core)")
print(df_cpu.to_markdown(index=True))

# --- Other metrics ---
df_other = df_stats[other_cols].T
df_other["normalized_mean"] = df_other["mean"] / num_cores
df_other["cov_%"] = round((df_other["std"] / df_other["mean"]) * 100, 2)
df_other = df_other[["mean", "normalized_mean", "std", "cov_%"]]
# df_other = df_other[["mean", "std", "cov_%"]]

# Save and print
df_other.to_csv("./bare_metal/other_variability.csv")
print("\n### Other Metrics")
print(df_other.to_markdown(index=True))
