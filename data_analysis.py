import os
import math
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter, label


TELEMETRY_ROOT = "telemetry data"
OUTPUT_CSV = "player_features_updated.csv"

TIMELINE_WITH_STATES = "unified_timeline_with_states.csv"

STATE_EXPLORATION = 0
STATE_PUZZLE = 1
STATE_MORAL = 2

NUM_BINS_X = 100
NUM_BINS_Z = 100

GAUSS_SIGMA = 2.0
HOTSPOT_THRESH_REL = 0.30
PEAK_MIN_REL = 0.20
NEIGHBORHOOD = 5

def load_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def ensure_time_sorted(df, time_col="time"):
    if time_col not in df.columns:
        raise ValueError(f"No time column'{time_col}'")
    return df.sort_values(time_col).reset_index(drop=True)


def preprocess_events(df_events):
    if df_events is None:
        return None

    df = df_events.copy()
    df.columns = [c.strip() for c in df.columns]

    df["eventType"] = df["eventType"].astype(str).str.strip()
    df["parameters"] = df.get("parameters", "").astype(str).fillna("")

    df["puzzleId"] = df["parameters"].str.extract(r"puzzleId=([^|]+)")[0]
    df["result"]   = df["parameters"].str.extract(r"result=([^|]+)")[0]
    df["ending"]   = df["parameters"].str.extract(r"ending=([^|]+)")[0]

    return df


def compute_heatmap_hotspot_features(df_pos):
    if df_pos is None or df_pos.empty:
        return math.nan, math.nan, math.nan

    if not all(col in df_pos.columns for col in ["time", "x", "z"]):
        return math.nan, math.nan, math.nan

    df = df_pos.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")

    df = df.dropna(subset=["time", "x", "z"]).sort_values("time").reset_index(drop=True)
    if len(df) < 5:
        return math.nan, math.nan, math.nan

    t = df["time"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)

    dt = np.diff(t, prepend=t[0])
    dt[0] = np.median(dt[1:]) if len(dt) > 2 else 0.0
    dt = np.clip(dt, 0.0, np.percentile(dt, 95))

    total_time = float(np.sum(dt))
    if total_time <= 0:
        return 0, 0.0, 0.0

    x_min, x_max = float(np.min(x)), float(np.max(x))
    z_min, z_max = float(np.min(z)), float(np.max(z))

    if abs(x_max - x_min) < 1e-9 or abs(z_max - z_min) < 1e-9:
        return 0, 0.0, total_time

    H, xedges, zedges = np.histogram2d(
        x, z,
        bins=[NUM_BINS_X, NUM_BINS_Z],
        range=[[x_min, x_max], [z_min, z_max]],
        weights=dt
    )

    Hs = gaussian_filter(H, sigma=GAUSS_SIGMA)

    peak_density_value = float(np.max(Hs)) if Hs.size else 0.0
    if peak_density_value <= 0:
        return 0, 0.0, 0.0

    neighborhood = (NEIGHBORHOOD, NEIGHBORHOOD)
    local_max = (Hs == maximum_filter(Hs, size=neighborhood))
    strong_peaks = Hs >= (PEAK_MIN_REL * peak_density_value)
    number_hotspots = int(np.sum(local_max & strong_peaks))

    hotspot_mask = Hs >= (HOTSPOT_THRESH_REL * peak_density_value)

    ix = np.clip(np.searchsorted(xedges, x, side="right") - 1, 0, NUM_BINS_X - 1)
    iz = np.clip(np.searchsorted(zedges, z, side="right") - 1, 0, NUM_BINS_Z - 1)

    time_in_hotspots = float(np.sum(dt[hotspot_mask[ix, iz]]))
    pct_time_in_hotspots = time_in_hotspots / total_time

    return number_hotspots, pct_time_in_hotspots, peak_density_value

def compute_state_features_from_sequence(states):
    if states is None or len(states) == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    states = np.asarray(states, dtype=int)

    n = len(states)

    pct_exploration = float(np.mean(states == STATE_EXPLORATION))
    pct_puzzle = float(np.mean(states == STATE_PUZZLE))
    pct_moral = float(np.mean(states == STATE_MORAL))

    counts = np.array([
        np.sum(states == STATE_EXPLORATION),
        np.sum(states == STATE_PUZZLE),
        np.sum(states == STATE_MORAL),
    ], dtype=float)

    probs = counts / counts.sum() if counts.sum() > 0 else np.array([0.0, 0.0, 0.0])
    probs = probs[probs > 0] 
    state_entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0

    run_lengths = []
    run_len = 1
    for i in range(1, n):
        if states[i] == states[i - 1]:
            run_len += 1
        else:
            run_lengths.append(run_len)
            run_len = 1
    run_lengths.append(run_len)

    return (
        pct_exploration,
        pct_puzzle,
        pct_moral,
        state_entropy,
    )
    
    
def compute_state_features_from_timeline_csv(timeline_path):
    if not os.path.exists(timeline_path):
        return (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    df = pd.read_csv(timeline_path)

    if "state_code" not in df.columns:
        return (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    states = pd.to_numeric(df["state_code"], errors="coerce").dropna().astype(int).tolist()

    return compute_state_features_from_sequence(states)


def compute_puzzle_stats(df_events, time_col="time"):
    if df_events is None or df_events.empty:
        return 0, 0, 0, math.nan, math.nan

    df = ensure_time_sorted(df_events, time_col)

    num_started = (df["eventType"] == "puzzle_start").sum()
    num_succeeded = ((df["eventType"] == "puzzle_end") & (df["result"] == "success")).sum()
    num_failed = ((df["eventType"] == "puzzle_end") & (df["result"] == "fail")).sum()

    puzzle_success_rate = (
        num_succeeded / num_started if num_started > 0 else math.nan
    )

    durations = []
    for pid, g in df.groupby("puzzleId"):
        if pd.isna(pid):
            continue

        g = g.sort_values(time_col)
        starts = g[g["eventType"] == "puzzle_start"]
        ends   = g[g["eventType"] == "puzzle_end"]

        if starts.empty or ends.empty:
            continue

        start_t = float(starts[time_col].iloc[0])
        end_t   = float(ends[time_col].iloc[0])

        if end_t > start_t:
            durations.append(end_t - start_t)

    avg_duration = float(np.mean(durations)) if durations else math.nan

    return num_started, num_succeeded, num_failed, avg_duration, puzzle_success_rate


def compute_total_game_time(df_events, df_positions, time_col="time"):
    times = []
    
    if df_events is not None and not df_events.empty and time_col in df_events.columns:
        ev = df_events
        starts = ev.loc[ev["eventType"] == "game_start", time_col]
        ends   = ev.loc[ev["eventType"] == "game_end", time_col]

        if not starts.empty and not ends.empty:
            return float(ends.iloc[-1] - starts.iloc[0])

        times.append(ev[time_col].min())
        times.append(ev[time_col].max())

    if df_positions is not None and not df_positions.empty:
        times.append(df_positions[time_col].min())
        times.append(df_positions[time_col].max())

    if len(times) < 2:
        return math.nan

    return float(max(times) - min(times))


def compute_morality_metrics(df_moral, value_col="value"):
    if df_moral is None or df_moral.empty:
        return math.nan, math.nan, math.nan, math.nan

    df = df_moral.sort_values("time").reset_index(drop=True)
    values = df[value_col].astype(float)

    final_morality = float(values.iloc[-1])

    mean_morality = float(values.mean())
    
    moral_std = float(values.std())
    moral_var = float(values.var())

    return mean_morality, final_morality, moral_std, moral_var


def get_ending_achieved(df_events):
    if df_events is None or df_events.empty:
        return None

    ev = df_events
    endings = ev.loc[ev["eventType"] == "game_end", "ending"]

    if not endings.empty:
        return endings.iloc[-1]

    vals = ev["ending"].dropna()
    if not vals.empty:
        return vals.iloc[-1]

    return None


def main():
    player_rows = []

    folders = sorted([
        f for f in os.listdir(TELEMETRY_ROOT)
        if os.path.isdir(os.path.join(TELEMETRY_ROOT, f))
    ])

    for i, folder_name in enumerate(folders, start=1):
        player_id = i
        folder_path = os.path.join(TELEMETRY_ROOT, folder_name)

        df_pos = load_csv_if_exists(os.path.join(folder_path, "positions.csv"))
        df_moral = load_csv_if_exists(os.path.join(folder_path, "morality_changes.csv"))
        df_events_raw = load_csv_if_exists(os.path.join(folder_path, "events.csv"))

        df_events = preprocess_events(df_events_raw) if df_events_raw is not None else None

        (mean_morality, final_morality, moral_std, moral_var) = compute_morality_metrics(df_moral)
        
        number_puzzles_started, number_puzzles_succeeded, number_puzzles_failed, avg_time_puzzles, puzzle_success_rate = compute_puzzle_stats(df_events)

        time_complete_game = compute_total_game_time(df_events, df_pos)

        ending_achieved = get_ending_achieved(df_events)

        timeline_path = os.path.join(folder_path, TIMELINE_WITH_STATES)

        (pct_exploration, pct_puzzle,pct_moral_decisions,state_entropy) = compute_state_features_from_timeline_csv(timeline_path)
        
        number_hotspots, pct_time_in_hotspots, peak_density_value = compute_heatmap_hotspot_features(df_pos)

        row = {
            "player_id": player_id,
            "mean_morality": mean_morality,
            "var_morality": moral_var,
            "std_morality": moral_std,
            "final_morality": final_morality,
            "npc_interactions": math.nan,
            "number_puzzles_started": number_puzzles_started,
            "number_puzzles_succeeded": number_puzzles_succeeded,
            "number_puzzles_failed": number_puzzles_failed,
            "puzzle_success_rate": puzzle_success_rate,
            "average_time_spent_puzzles": avg_time_puzzles,
            "time_complete_game": time_complete_game,
            "ending_achieved": ending_achieved,
            "%_exploration": pct_exploration,
            "%_puzzle": pct_puzzle,
            "%_moral_decisions": pct_moral_decisions,
            "state_entropy": state_entropy,
            "number_hotspots": number_hotspots,
            "pct_time_in_hotspots": pct_time_in_hotspots,
            "peak_density_value": peak_density_value,
        }

        player_rows.append(row)

    df_out = pd.DataFrame(player_rows)

    df_out = df_out.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, float) else x)
    
    df_out.to_csv(OUTPUT_CSV, index=False, sep=";")

if __name__ == "__main__":
    main()