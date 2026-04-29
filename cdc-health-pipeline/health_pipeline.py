"""
health_pipeline.py
CS469 Big Data Analytics — Unit 5 Individual Project (Final IP)
Author: Franklin Woodard

Full data pipeline on CDC PLACES County Health Data (2022 release):
  1. Ingest      — load CSV from local path
  2. Storage     — structured pandas DataFrame
  3. Cleansing   — handle missing values, drop CI columns, normalize types
  4. Preprocessing — derive composite health index, state aggregations
  5. Utilization — descriptive report + visualizations

Dataset: CDC PLACES County Data (GIS Friendly Format) 2022
Source:  https://data.cdc.gov / data.gov
"""

import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH   = "PLACES__County_Data__GIS_Friendly_Format___2022_release.csv"
OUTPUT_DIR = Path("pipeline_output")
OUTPUT_DIR.mkdir(exist_ok=True)

SEP = "─" * 65

# ── HEALTH METRIC COLUMNS (CrudePrev only — drop CI and Adj columns) ──────────
HEALTH_METRICS = {
    'ACCESS2':   'No Health Insurance (%)',
    'ARTHRITIS':  'Arthritis (%)',
    'BINGE':      'Binge Drinking (%)',
    'BPHIGH':     'High Blood Pressure (%)',
    'BPMED':      'BP Medication Use (%)',
    'CANCER':     'Cancer (non-skin) (%)',
    'CASTHMA':    'Current Asthma (%)',
    'CHD':        'Coronary Heart Disease (%)',
    'CHECKUP':    'Annual Checkup (%)',
    'CHOLSCREEN': 'Cholesterol Screening (%)',
    'COLON_SCREEN':'Colorectal Screening (%)',
    'COPD':       'COPD (%)',
    'COREM':      'Core Preventive (Men) (%)',
    'COREW':      'Core Preventive (Women) (%)',
    'CSMOKING':   'Current Smoking (%)',
    'DENTAL':     'Dental Visit (%)',
    'DEPRESSION': 'Depression (%)',
    'DIABETES':   'Diabetes (%)',
    'HIGHCHOL':   'High Cholesterol (%)',
    'KIDNEY':     'Chronic Kidney Disease (%)',
    'LPA':        'Physical Inactivity (%)',
    'MAMMOUSE':   'Mammography Use (%)',
    'MHLTH':      'Poor Mental Health (%)',
    'OBESITY':    'Obesity (%)',
    'PHLTH':      'Poor Physical Health (%)',
    'SLEEP':      'Insufficient Sleep (%)',
    'STROKE':     'Stroke (%)',
    'TEETHLOST':  'All Teeth Lost (%)',
}

CRUDE_COLS = [f"{k}_CrudePrev" for k in HEALTH_METRICS.keys()]

# Negative health indicators — higher = worse
NEGATIVE = ['ACCESS2','ARTHRITIS','BINGE','BPHIGH','CANCER','CASTHMA',
            'CHD','COPD','CSMOKING','DEPRESSION','DIABETES','HIGHCHOL',
            'KIDNEY','LPA','MHLTH','OBESITY','PHLTH','SLEEP','STROKE','TEETHLOST']

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"{'CDC PLACES COUNTY HEALTH DATA PIPELINE':^65}")
print(f"{'CS469 Big Data Analytics — Unit 5 Final IP':^65}")
print(SEP)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — INGEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] INGEST")
print(SEP)

raw = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  Source       : {CSV_PATH}")
print(f"  Format       : CSV")
print(f"  Raw rows     : {len(raw):,}")
print(f"  Raw columns  : {len(raw.columns)}")
print(f"  States       : {raw['StateAbbr'].nunique()}")
print(f"  Counties     : {len(raw):,}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — STORAGE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] STORAGE")
print(SEP)

# Select only the columns we need
id_cols   = ['StateAbbr', 'StateDesc', 'CountyName', 'CountyFIPS', 'TotalPopulation']
keep_cols = id_cols + [c for c in CRUDE_COLS if c in raw.columns]
df = raw[keep_cols].copy()

print(f"  Columns retained : {len(df.columns)} (id + {len(CRUDE_COLS)} health metrics)")
print(f"  Memory usage     : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"  Data structure   : pandas DataFrame")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CLEANSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] CLEANSING")
print(SEP)

rows_before = len(df)

# Coerce health metric columns to numeric (some may have been read as string)
for col in CRUDE_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Standardize county name casing
df['CountyName'] = df['CountyName'].str.strip().str.title()
df['StateAbbr']  = df['StateAbbr'].str.strip().str.upper()

# Coerce population to int
df['TotalPopulation'] = pd.to_numeric(df['TotalPopulation'], errors='coerce')

# Drop rows missing population or state
df = df.dropna(subset=['TotalPopulation', 'StateAbbr'])

# Report missing values per health metric
missing = {c: df[c].isna().sum() for c in CRUDE_COLS if c in df.columns}
cols_with_missing = {k: v for k, v in missing.items() if v > 0}

print(f"  Rows before cleansing : {rows_before:,}")
print(f"  Rows after cleansing  : {len(df):,}")
print(f"  Rows dropped          : {rows_before - len(df)}")
if cols_with_missing:
    print(f"  Columns with NaN      : {len(cols_with_missing)}")
    for col, n in list(cols_with_missing.items())[:5]:
        print(f"    {col}: {n} missing")
else:
    print(f"  Missing values        : None in health metric columns")

# Fill remaining NaN in health metrics with column median (preserves distributions)
for col in CRUDE_COLS:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] PREPROCESSING")
print(SEP)

# Rename crude prevalence columns to readable names
rename_map = {f"{k}_CrudePrev": v for k, v in HEALTH_METRICS.items()
              if f"{k}_CrudePrev" in df.columns}
df = df.rename(columns=rename_map)
metric_cols = list(rename_map.values())

# Composite Health Burden Index:
# Average of normalized negative-indicator columns (higher = worse health burden)
neg_readable = [HEALTH_METRICS[k] for k in NEGATIVE if HEALTH_METRICS[k] in metric_cols]

def minmax(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn) if mx > mn else series * 0

df['HealthBurdenIndex'] = df[neg_readable].apply(minmax).mean(axis=1).round(4)

# State-level aggregations (population-weighted)
def wavg(group, col):
    """Population-weighted average."""
    return np.average(group[col], weights=group['TotalPopulation'])

state_agg = df.groupby('StateAbbr').apply(
    lambda g: pd.Series({
        'TotalPopulation'    : g['TotalPopulation'].sum(),
        'CountyCount'        : len(g),
        'AvgBurdenIndex'     : wavg(g, 'HealthBurdenIndex'),
        'NoInsurance'        : wavg(g, 'No Health Insurance (%)'),
        'Obesity'            : wavg(g, 'Obesity (%)'),
        'Smoking'            : wavg(g, 'Current Smoking (%)'),
        'Depression'         : wavg(g, 'Depression (%)'),
        'Diabetes'           : wavg(g, 'Diabetes (%)'),
        'PhysicalInactivity' : wavg(g, 'Physical Inactivity (%)'),
        'PoorMentalHealth'   : wavg(g, 'Poor Mental Health (%)'),
    })
).reset_index().sort_values('AvgBurdenIndex', ascending=False)

print(f"  Composite Health Burden Index built from {len(neg_readable)} negative indicators")
print(f"  State aggregations computed for {len(state_agg)} states")
print(f"  Population-weighted averages used to account for county size differences")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — UTILIZATION: DESCRIPTIVE REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] UTILIZATION — Descriptive Report")
print(SEP)

# ── NATIONAL SUMMARY ─────────────────────────────────────────────────────────
total_pop = df['TotalPopulation'].sum()
print(f"\n  NATIONAL SUMMARY ({len(df):,} counties, {df['StateAbbr'].nunique()} states)")
print(f"  {'─'*50}")
print(f"  Total covered population : {total_pop:,.0f}")

for metric in ['No Health Insurance (%)', 'Obesity (%)', 'Current Smoking (%)',
               'Depression (%)', 'Diabetes (%)', 'Poor Mental Health (%)']:
    if metric in df.columns:
        wm = np.average(df[metric], weights=df['TotalPopulation'])
        print(f"  {metric:<35}: {wm:>5.1f}%")

# ── TOP / BOTTOM STATES BY BURDEN INDEX ──────────────────────────────────────
print(f"\n  TOP 10 HIGHEST BURDEN STATES (Health Burden Index)")
print(f"  {'State':<8} {'Burden Index':>13} {'No Insurance':>13} {'Obesity':>9} {'Smoking':>9}")
for _, row in state_agg.head(10).iterrows():
    print(f"  {row['StateAbbr']:<8} {row['AvgBurdenIndex']:>13.4f} "
          f"{row['NoInsurance']:>12.1f}% {row['Obesity']:>8.1f}% {row['Smoking']:>8.1f}%")

print(f"\n  TOP 10 LOWEST BURDEN STATES (Healthiest)")
print(f"  {'State':<8} {'Burden Index':>13} {'No Insurance':>13} {'Obesity':>9} {'Smoking':>9}")
for _, row in state_agg.tail(10).sort_values('AvgBurdenIndex').iterrows():
    print(f"  {row['StateAbbr']:<8} {row['AvgBurdenIndex']:>13.4f} "
          f"{row['NoInsurance']:>12.1f}% {row['Obesity']:>8.1f}% {row['Smoking']:>8.1f}%")

# ── TOP 15 HIGHEST BURDEN COUNTIES ───────────────────────────────────────────
print(f"\n  TOP 15 HIGHEST BURDEN COUNTIES")
top_counties = df.nlargest(15, 'HealthBurdenIndex')[
    ['CountyName', 'StateAbbr', 'TotalPopulation', 'HealthBurdenIndex',
     'No Health Insurance (%)', 'Obesity (%)', 'Diabetes (%)']
]
print(f"  {'County':<25} {'ST':>3} {'Population':>12} {'Burden':>8} {'Uninsured':>10} {'Obesity':>8} {'Diabetes':>9}")
for _, row in top_counties.iterrows():
    print(f"  {row['CountyName']:<25} {row['StateAbbr']:>3} "
          f"{row['TotalPopulation']:>12,.0f} {row['HealthBurdenIndex']:>8.4f} "
          f"{row['No Health Insurance (%)']:>9.1f}% {row['Obesity (%)']:>7.1f}% "
          f"{row['Diabetes (%)']:>8.1f}%")

# ── SAVE PROCESSED DATASET ───────────────────────────────────────────────────
out_csv = OUTPUT_DIR / "counties_processed.csv"
df.to_csv(out_csv, index=False)
state_csv = OUTPUT_DIR / "states_summary.csv"
state_agg.to_csv(state_csv, index=False)
print(f"\n  Processed county data saved : {out_csv}")
print(f"  State summary saved         : {state_csv}")

# ── VISUALIZATIONS ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('CDC PLACES County Health Data — National Overview (2022)',
             fontsize=15, fontweight='bold', y=0.98)

# Plot 1: Top 15 states by burden index
ax1 = axes[0, 0]
top15 = state_agg.head(15).sort_values('AvgBurdenIndex')
bars = ax1.barh(top15['StateAbbr'], top15['AvgBurdenIndex'],
                color=plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, 15)))
ax1.set_xlabel('Health Burden Index (higher = worse)')
ax1.set_title('Top 15 Highest Burden States')
ax1.set_xlim(0, top15['AvgBurdenIndex'].max() * 1.15)

# Plot 2: Obesity vs Diabetes scatter
ax2 = axes[0, 1]
sc = ax2.scatter(df['Obesity (%)'], df['Diabetes (%)'],
                 c=df['HealthBurdenIndex'], cmap='RdYlGn_r',
                 alpha=0.5, s=15, linewidths=0)
plt.colorbar(sc, ax=ax2, label='Burden Index')
ax2.set_xlabel('Obesity (%)')
ax2.set_ylabel('Diabetes (%)')
ax2.set_title('Obesity vs. Diabetes by County\n(color = Health Burden Index)')

# Plot 3: Distribution of Health Burden Index
ax3 = axes[1, 0]
ax3.hist(df['HealthBurdenIndex'], bins=50, color='steelblue',
         edgecolor='white', linewidth=0.4)
ax3.axvline(df['HealthBurdenIndex'].mean(), color='red',
            linestyle='--', linewidth=1.5, label=f"Mean: {df['HealthBurdenIndex'].mean():.3f}")
ax3.set_xlabel('Health Burden Index')
ax3.set_ylabel('Number of Counties')
ax3.set_title('Distribution of Health Burden Index\nAcross All Counties')
ax3.legend()

# Plot 4: Top 10 metrics by national average
ax4 = axes[1, 1]
national_avgs = {m: np.average(df[m], weights=df['TotalPopulation'])
                 for m in metric_cols if m in df.columns}
top10_metrics = sorted(national_avgs.items(), key=lambda x: x[1], reverse=True)[:10]
labels = [x[0].replace(' (%)', '').replace(' ', '\n') for x, _ in top10_metrics]
values = [v for _, v in top10_metrics]
ax4.barh(labels, values, color='steelblue')
ax4.set_xlabel('Prevalence (%)')
ax4.set_title('Top 10 Health Metrics\nby National Weighted Average')
ax4.invert_yaxis()

plt.tight_layout()
chart_path = OUTPUT_DIR / "health_pipeline_report.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Charts saved : {chart_path}")

print(f"\n{SEP}")
print("Pipeline complete.")
print(SEP)
