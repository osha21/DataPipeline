import os
import pandas as pd
import numpy as np

TELEMETRY_ROOT = "telemetry data"
UNIFIED_FILENAME = "unified_timeline.csv"

OUT_TIMELINE_NAME = "unified_timeline_with_states.csv"

STATE_EXPLORATION = 0
STATE_PUZZLE = 1
STATE_MORAL = 2

LABELS = {
    STATE_EXPLORATION: "EXPLORATION",
    STATE_PUZZLE: "PUZZLE",
    STATE_MORAL: "MORAL_DECISION"
}

def has_token(events_str, token):
    s = str(events_str).strip()
    parts = [p.strip() for p in s.split("|") if p.strip()]
    return token in parts

def map_states(df, tol=1e-9):
    df = df.sort_values("time").reset_index(drop=True).copy()

    if "events_types_window" in df.columns:
        ev_col = "events_types_window"
    elif "events_details_window" in df.columns:
        ev_col = "events_details_window"

    df["morality_at_time"] = pd.to_numeric(df["morality_at_time"], errors="coerce")

    in_puzzle = False
    states = []

    prev_m = None

    for row in df.iterrows():
        ev = row.get(ev_col, "") if ev_col else ""
        start_here = has_token(ev, "puzzle_start") if ev_col else False
        end_here   = has_token(ev, "puzzle_end") if ev_col else False

        if start_here:
            in_puzzle = True

        m = row["morality_at_time"]
        moral_changed = False

        if pd.notna(m):
            if prev_m is not None and pd.notna(prev_m):
                moral_changed = abs(float(m) - float(prev_m)) > tol
            prev_m = m
        if moral_changed:
            state = STATE_MORAL
        elif in_puzzle:
            state = STATE_PUZZLE
        else:
            state = STATE_EXPLORATION

        states.append(state)

        if end_here:
            in_puzzle = False

    df["state_code"] = states
    df["state_label"] = [LABELS[s] for s in states]
    return df

def main():
    folders = sorted([f for f in os.listdir(TELEMETRY_ROOT) if os.path.isdir(os.path.join(TELEMETRY_ROOT, f))])

    for folder_name in enumerate(folders, start=1):
        folder_path = os.path.join(TELEMETRY_ROOT, folder_name)
        unified_path = os.path.join(folder_path, UNIFIED_FILENAME)

        df = pd.read_csv(unified_path)
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])

        df_out = map_states(df)

        out_timeline = os.path.join(folder_path, OUT_TIMELINE_NAME)
        df_out.to_csv(out_timeline, index=False)

if __name__ == "__main__":
    main()