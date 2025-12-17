import os
import math
import numpy as np
import pandas as pd

TELEMETRY_ROOT = "telemetry data"
OUTPUT_PER_PLAYER = "unified_timeline.csv"

def load_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def ensure_time_sorted(df, time_col="time"):
    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    return df.sort_values(time_col).reset_index(drop=True)

def preprocess_events(df_events):
    df = df_events.copy()
    df.columns = [c.strip() for c in df.columns]

    df["eventType"] = df["eventType"].astype(str).str.strip()
    if "parameters" not in df.columns:
        df["parameters"] = ""
    df["parameters"] = df["parameters"].astype(str).fillna("")

    df["puzzleId"] = df["parameters"].str.extract(r"puzzleId=([^|]+)")[0]
    df["result"]   = df["parameters"].str.extract(r"result=([^|]+)")[0]
    df["ending"]   = df["parameters"].str.extract(r"ending=([^|]+)")[0]

    return df


def attach_morality_asof(df_pos, df_moral, time_col="time", moral_value_col="value"):
    df_pos = df_pos.copy()

    df_moral = df_moral.copy()
    df_moral.columns = [c.strip() for c in df_moral.columns]

    if moral_value_col not in df_moral.columns:
        candidates = [c for c in df_moral.columns if c.lower() in ("value", "morality", "moralityvalue")]
        moral_value_col = candidates[0]

    df_moral[time_col] = pd.to_numeric(df_moral[time_col], errors="coerce")
    df_moral[moral_value_col] = pd.to_numeric(df_moral[moral_value_col], errors="coerce")
    df_moral = df_moral.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    out = pd.merge_asof(
        df_pos.sort_values(time_col),
        df_moral[[time_col, moral_value_col]].sort_values(time_col),
        on=time_col,
        direction="backward"
    )

    out = out.rename(columns={moral_value_col: "morality_at_time"})
    return out.reset_index(drop=True)

def attach_events_between_positions(df_pos, df_events, time_col="time"):
    df_pos = df_pos.copy()

    if df_events is None or df_events.empty:
        df_pos["events_types_window"] = ""
        df_pos["events_details_window"] = ""
        return df_pos

    df_events = df_events.copy()
    df_events = ensure_time_sorted(df_events, time_col=time_col)

    ev_times = df_events[time_col].to_numpy()
    pos_times = df_pos[time_col].to_numpy()

    event_types_col = df_events["eventType"].astype(str).to_numpy()
    def build_detail_row(i):
        et = df_events.iloc[i]["eventType"]
        pid = df_events.iloc[i].get("puzzleId", "")
        res = df_events.iloc[i].get("result", "")
        end = df_events.iloc[i].get("ending", "")
        parts = [f"type={et}"]
        if isinstance(pid, str) and pid and pid != "nan":
            parts.append(f"puzzleId={pid}")
        if isinstance(res, str) and res and res != "nan":
            parts.append(f"result={res}")
        if isinstance(end, str) and end and end != "nan":
            parts.append(f"ending={end}")
        return ",".join(parts)

    event_details = [build_detail_row(i) for i in range(len(df_events))]

    types_out = []
    details_out = []

    prev_t = -math.inf
    for t in pos_times:
        left = np.searchsorted(ev_times, prev_t, side="right")
        right = np.searchsorted(ev_times, t, side="right")

        if right > left:
            window_types = event_types_col[left:right]
            window_details = event_details[left:right]
            types_out.append("|".join(window_types))
            details_out.append("|".join(window_details))
        else:
            types_out.append("")
            details_out.append("")

        prev_t = t

    df_pos["events_types_window"] = types_out
    df_pos["events_details_window"] = details_out
    return df_pos

def build_unified_timeline(df_pos, df_events, df_moral, time_col="time"):
    df_pos = ensure_time_sorted(df_pos, time_col=time_col)

    df = attach_morality_asof(df_pos, df_moral, time_col=time_col, moral_value_col="value")

    df_events = preprocess_events(df_events) if df_events is not None else None
    df = attach_events_between_positions(df, df_events, time_col=time_col)

    return df

def main():
    all_rows = []

    folders = sorted([
        f for f in os.listdir(TELEMETRY_ROOT)
        if os.path.isdir(os.path.join(TELEMETRY_ROOT, f))
    ])

    for player_id, folder_name in enumerate(folders, start=1):
        folder_path = os.path.join(TELEMETRY_ROOT, folder_name)

        df_pos = load_csv_if_exists(os.path.join(folder_path, "positions.csv"))
        df_events = load_csv_if_exists(os.path.join(folder_path, "events.csv"))
        df_moral = load_csv_if_exists(os.path.join(folder_path, "morality_changes.csv"))

        unified = build_unified_timeline(df_pos, df_events, df_moral)

        unified.insert(0, "player_id", player_id)

        out_path = os.path.join(folder_path, OUTPUT_PER_PLAYER)
        unified.to_csv(out_path, index=False)

        all_rows.append(unified)

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv(OUTPUT_ALL, index=False)

if __name__ == "__main__":
    main()