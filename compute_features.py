#!/usr/bin/env python3
"""
Compute time-aware, strictly-prior features for matches in transformed.csv
and write transformed_augmented.csv.
Features added (prefix p1_/p2_ where appropriate):
- rank_diff, rank_diff_abs, rank_rel
- p1_win_streak, p2_win_streak
- p1_lose_streak, p2_lose_streak
- p1_surface_wr, p2_surface_wr
- p1_titles, p2_titles
- age_diff, height_diff
- win_streak_diff, lose_streak_diff, surface_wr_diff, titles_diff

The script tries to be defensive about column names.
"""
import pandas as pd
import numpy as np
import os

IN = 'transformed.csv'
OUT = 'transformed_augmented.csv'

if not os.path.exists(IN):
    raise SystemExit(f"Input file '{IN}' not found in cwd={os.getcwd()}")

print('Loading', IN)
df = pd.read_csv(IN, dtype=str)
# Keep a copy of original columns
orig_cols = df.columns.tolist()

# Parse tourney_date in YYYYMMDD format
if 'tourney_date' not in df.columns:
    raise SystemExit("Expected column 'tourney_date' in transformed.csv")

# Normalize common numeric/string artifacts (like '20150104.0') by removing trailing .0
# then keep only digits and extract an 8-digit YYYYMMDD string before parsing.
_raw_dates = df['tourney_date'].astype(str).fillna('')
_clean = (_raw_dates
          .str.replace(r"\.0$", "", regex=True)
          .str.replace(r"[^0-9]", "", regex=True)
          .str.extract(r"(\d{8})", expand=False)
         )
df['tourney_date'] = pd.to_datetime(_clean, format='%Y%m%d', errors='coerce')

# Helper to pick columns robustly
def pick(df, opts):
    for c in opts:
        if c in df.columns:
            return c
    return None

p1_id_col = pick(df, ['player_1_id','p1_id','player1_id'])
p2_id_col = pick(df, ['player_2_id','p2_id','player2_id'])
if p1_id_col is None or p2_id_col is None:
    raise SystemExit('Could not find player id columns (player_1_id / player_2_id)')

p1_name_col = pick(df, ['player_1_name','player1_name','p1_name'])
p2_name_col = pick(df, ['player_2_name','player2_name','p2_name'])

# rank columns (optional)
p1_rank_col = pick(df, ['player_1_rank','p1_rank','player1_rank'])
p2_rank_col = pick(df, ['player_2_rank','p2_rank','player2_rank'])

# age / height columns (optional)
p1_age_col = pick(df, ['player_1_age','p1_age','player1_age'])
p2_age_col = pick(df, ['player_2_age','p2_age','player2_age'])

p1_ht_col = pick(df, ['player_1_ht','p1_ht','player1_ht'])
p2_ht_col = pick(df, ['player_2_ht','p2_ht','player2_ht'])

# essential columns: result, surface, round
result_col = pick(df, ['result'])
surface_col = pick(df, ['surface'])
round_col = pick(df, ['round'])

# Normalize id columns to numeric keys
df['p1_key'] = pd.to_numeric(df[p1_id_col], errors='coerce').fillna(-1).astype(int)
    
df['p2_key'] = pd.to_numeric(df[p2_id_col], errors='coerce').fillna(-1).astype(int)

# Build long-form table: one row per player per match, with win indicator
p1 = df[['tourney_date', surface_col, 'p1_key']].copy()
p1 = p1.rename(columns={'p1_key':'player_key', surface_col:'surface'})
p1['win'] = pd.to_numeric(df[result_col], errors='coerce').fillna(0).astype(int)

p2 = df[['tourney_date', surface_col, 'p2_key']].copy()
p2 = p2.rename(columns={'p2_key':'player_key', surface_col:'surface'})
# p2 win is opposite of result (assuming result==1 means player_1 won)
p2['win'] = 1 - pd.to_numeric(df[result_col], errors='coerce').fillna(0).astype(int)

long = pd.concat([p1, p2], ignore_index=True, sort=False)
# sort by player_key and date
long = long.sort_values(['player_key','tourney_date']).reset_index(drop=True)

# compute streaks and surface stats per player

def compute_player_features(g):
    g = g.sort_values('tourney_date').reset_index(drop=True)
    wins = g['win'].to_numpy(dtype=int)
    n = len(wins)
    wst = np.zeros(n, dtype=int)
    lst = np.zeros(n, dtype=int)
    w = l = 0
    for i in range(n):
        wst[i] = w
        lst[i] = l
        if wins[i] == 1:
            w += 1
            l = 0
        else:
            l += 1
            w = 0
    g['win_streak'] = wst
    g['lose_streak'] = lst
    # surface prior matches: cumcount gives number of previous matches on that surface
    g['surface_matches'] = g.groupby('surface').cumcount()
    # surface prior wins: cumulative wins per surface, shifted so current match excluded
    g['surface_wins'] = g.groupby('surface')['win'].transform(lambda s: s.cumsum().shift(fill_value=0))
    # win rate on surface: when surface_matches == 0 -> set 0
    g['surface_wr'] = (g['surface_wins'] / g['surface_matches']).replace([np.inf, np.nan], 0)
    return g

long = long.groupby('player_key', group_keys=False).apply(compute_player_features)
long = long.sort_values(['player_key','tourney_date']).reset_index(drop=True)

# Titles: find final winners (if round column exists), else skip
if round_col is not None:
    df['is_final'] = df[round_col].astype(str).isin(['F','Final','f','final'])
    finals = df[df['is_final']].copy()
    if not finals.empty:
        finals['winner_key'] = np.where(pd.to_numeric(finals[result_col], errors='coerce')==1,
                                        pd.to_numeric(finals[p1_id_col], errors='coerce').fillna(-1).astype(int),
                                        pd.to_numeric(finals[p2_id_col], errors='coerce').fillna(-1).astype(int))
        titles = finals[['winner_key','tourney_date']].dropna().rename(columns={'winner_key':'player_key'})
        titles = titles.sort_values(['player_key','tourney_date']).reset_index(drop=True)
        titles['titles_before_at_title'] = titles.groupby('player_key').cumcount()
        # map titles_before to long rows using merge-as-of semantics achieved via dict mapping on kdate
        titles['tourney_date_k'] = titles['tourney_date'].dt.strftime('%Y%m%d')
        titles['kdate'] = titles['player_key'].astype(str) + '|' + titles['tourney_date_k']
        # For each title occurrence, titles_before_at_title is the count BEFORE that title (i.e., previous titles)
        # We'll build a map of kdate -> titles_before_at_title and later map to long rows by matching exact kdate
        titles_map = dict(zip(titles['kdate'], titles['titles_before_at_title']))
    else:
        titles_map = {}
else:
    titles_map = {}

# Prepare features mapping using kdate keys
# features needed: win_streak, lose_streak, surface_wr, titles_before
long['tourney_date_k'] = long['tourney_date'].dt.strftime('%Y%m%d')
long['kdate'] = long['player_key'].astype(str) + '|' + long['tourney_date_k']

# Build mapping dictionaries for per-player-date features (these represent the value BEFORE the current match)
feat_cols = ['win_streak','lose_streak','surface_wr']
maps = {}
for col in feat_cols:
    maps[col] = dict(zip(long['kdate'], long[col]))
# titles: we need the number of titles before the current match date.
# titles_map currently has entries at title dates mapping to titles_before_at_title (titles already held before that title)
# For long rows on non-title dates, we want the latest titles_before value strictly before the date.
# A memory-light approach: build per-player cumulative titles series and then map via kdate on long.

# Build per-player cumulative titles by merging titles onto long using groupby cumcount approach
if titles_map:
    # We'll create a titles_before column by looking up for exact kdate in titles_map; where missing, we'll set 0.
    long['titles_before'] = long['kdate'].map(titles_map).fillna(0).astype(int)
    # The above gives titles_before at dates that are exactly title dates. For prior matches we need the most recent prior title count.
    # To fill this in, we'll forward-fill per player the titles_before value.
    long['titles_before'] = long.groupby('player_key')['titles_before'].ffill().fillna(0).astype(int)
else:
    long['titles_before'] = 0

# Now create a features table keyed by player_key and tourney_date_k
features = long.set_index('kdate')[['win_streak','lose_streak','surface_wr','titles_before']]

# Prepare match-level keys to map per-player features for p1/p2
# create tourney_date_k in df
df['tourney_date_k'] = df['tourney_date'].dt.strftime('%Y%m%d')
df['p1_kdate'] = df['p1_key'].astype(str) + '|' + df['tourney_date_k']
df['p2_kdate'] = df['p2_key'].astype(str) + '|' + df['tourney_date_k']

# Map each feature for p1 and p2 using the features dict
for col in ['win_streak','lose_streak','surface_wr','titles_before']:
    fmap = features[col].to_dict()
    # normalize target column name: use 'titles' as the final column name (not 'titles_before')
    target = 'titles' if col == 'titles_before' else col
    df[f'p1_{target}'] = df['p1_kdate'].map(fmap).fillna(0)
    df[f'p2_{target}'] = df['p2_kdate'].map(fmap).fillna(0)

# Fix dtypes: streaks and titles should be integers, surface_wr a float
for streak_col in ['win_streak','lose_streak']:
    for side in ['p1_','p2_']:
        c = side + streak_col
        if c in df.columns:
            # map produced numeric-ish values; coerce to int safely
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

for side in ['p1_','p2_']:
    # titles
    tcol = side + 'titles'
    if tcol in df.columns:
        df[tcol] = pd.to_numeric(df[tcol], errors='coerce').fillna(0).astype(int)
    # surface winrate
    sw = side + 'surface_wr'
    if sw in df.columns:
        df[sw] = pd.to_numeric(df[sw], errors='coerce').fillna(0.0)

# Compute requested derived features
# rank diff
if p1_rank_col and p2_rank_col:
    df['p1_rank'] = pd.to_numeric(df[p1_rank_col], errors='coerce')
    df['p2_rank'] = pd.to_numeric(df[p2_rank_col], errors='coerce')
    df['rank_diff'] = df['p2_rank'] - df['p1_rank']
    df['rank_diff_abs'] = df['rank_diff'].abs()
    # relative difference: (p2-p1) / max(1, p1+p2)
    df['rank_rel'] = df['rank_diff'] / (df[['p1_rank','p2_rank']].sum(axis=1).replace(0,np.nan)).abs()
else:
    df['rank_diff'] = np.nan
    df['rank_diff_abs'] = np.nan
    df['rank_rel'] = np.nan

# age diff and height diff
if p1_age_col and p2_age_col:
    df['p1_age'] = pd.to_numeric(df[p1_age_col], errors='coerce')
    df['p2_age'] = pd.to_numeric(df[p2_age_col], errors='coerce')
    df['age_diff'] = df['p1_age'] - df['p2_age']
else:
    df['age_diff'] = np.nan

if p1_ht_col and p2_ht_col:
    df['p1_ht'] = pd.to_numeric(df[p1_ht_col], errors='coerce')
    df['p2_ht'] = pd.to_numeric(df[p2_ht_col], errors='coerce')
    df['height_diff'] = df['p1_ht'] - df['p2_ht']
else:
    df['height_diff'] = np.nan

# differences between players for features
df['win_streak_diff'] = df['p1_win_streak'] - df['p2_win_streak']
df['lose_streak_diff'] = df['p1_lose_streak'] - df['p2_lose_streak']
df['surface_wr_diff'] = df['p1_surface_wr'] - df['p2_surface_wr']
df['titles_diff'] = df['p1_titles'] - df['p2_titles']

# Clean helper cols
for c in ['tourney_date_k','p1_kdate','p2_kdate','p1_key','p2_key']:
    if c in df.columns:
        df = df.drop(columns=[c])

# Save result
print('Writing', OUT)
# Keep original column order first, then new columns
new_cols = [c for c in df.columns if c not in orig_cols]
out_cols = orig_cols + new_cols
# Ensure we don't include duplicate columns
out_cols = [*dict.fromkeys(out_cols)]

# Convert tourney_date back to YYYYMMDD string to match original format
if pd.api.types.is_datetime64_any_dtype(df['tourney_date']):
    df['tourney_date'] = df['tourney_date'].dt.strftime('%Y%m%d')

# === Type conversions for modeling readiness ===
# 1) Convert object columns to categorical codes (cat.codes)
obj_cols = df.select_dtypes(include=['object']).columns.tolist()
for c in obj_cols:
    try:
        df[c] = df[c].astype('category').cat.codes
    except Exception:
        # fallback: keep as-is
        pass

# 2) For float columns that are actually integers, convert to nullable Int64
float_cols = df.select_dtypes(include=['float']).columns.tolist()
for c in float_cols:
    s = pd.to_numeric(df[c], errors='coerce')
    non_na = s.dropna()
    if non_na.empty:
        # no info, convert to Int64 with NA
        df[c] = s.astype('Int64')
        continue
    # if all non-NA values are integers (no fractional part)
    if (non_na % 1 == 0).all():
        df[c] = s.round(0).astype('Int64')
    else:
        # keep as float but ensure numeric dtype
        df[c] = s.astype(float)

# 3) Ensure newly created streak/titles columns use safe integer types
for col in ['p1_win_streak','p2_win_streak','p1_lose_streak','p2_lose_streak','p1_titles','p2_titles']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('Int64')

# 4) Ensure surface wr columns are float
for col in ['p1_surface_wr','p2_surface_wr']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)

df.to_csv(OUT, index=False, columns=out_cols)
print('Done. Output saved to', OUT)
