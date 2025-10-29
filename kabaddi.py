# kabaddi.py
import os
import time
from datetime import datetime
import pandas as pd
import streamlit as st

# -------- CONFIG ----------
DATA_DIR = "folder_path"
os.makedirs(DATA_DIR, exist_ok=True)

TOURNAMENTS_CSV = os.path.join(DATA_DIR, "kabaddi_tournaments.csv")
TEAMS_CSV = os.path.join(DATA_DIR, "kabaddi_teams.csv")
PLAYERS_CSV = os.path.join(DATA_DIR, "kabaddi_players.csv")
MATCHES_CSV = os.path.join(DATA_DIR, "kabaddi_matches.csv")
SCORES_CSV = os.path.join(DATA_DIR, "kabaddi_scores.csv")
MATCH_STATE_CSV = os.path.join(DATA_DIR, "kabaddi_match_state.csv")

# Kabaddi rules
HALF_SECONDS = 20 * 60        # 20 minutes per half
TOTAL_SECONDS = HALF_SECONDS * 2
MAX_TIMEOUTS_PER_TEAM = 2

# Headers according to your specified schema
TOURNAMENTS_HDR = ["tournament_id", "tournament_name", "start_date", "end_date", "location"]
TEAMS_HDR = ["team_id", "team_name", "tournament_id"]  # if you want 'logo' you can add it later
PLAYERS_HDR = ["player_id", "player_name", "team_id"]
MATCHES_HDR = ["match_id", "tournament_id", "team1_id", "team2_id", "match_date", "venue"]
SCORES_HDR = ["match_id", "minute", "event_type", "player_id", "team_id", "points", "details", "timestamp"]
MATCH_STATE_HDR = ["match_id", "half", "start_time_iso", "accumulated_seconds", "is_running", "team1_timeouts", "team2_timeouts"]

# --------- Utility CSV helpers ----------
def ensure_csv(path, cols):
    """Create CSV with headers if missing or empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=cols).to_csv(path, index=False)

def load_csv(path, cols=None):
    """Load CSV; if file missing or empty create it with headers and return empty DF."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        if cols is None:
            return pd.DataFrame()
        else:
            pd.DataFrame(columns=cols).to_csv(path, index=False)
            return pd.DataFrame(columns=cols)
    return pd.read_csv(path)

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

# Ensure all files exist with the correct headers
ensure_csv(TOURNAMENTS_CSV, TOURNAMENTS_HDR)
ensure_csv(TEAMS_CSV, TEAMS_HDR)
ensure_csv(PLAYERS_CSV, PLAYERS_HDR)
ensure_csv(MATCHES_CSV, MATCHES_HDR)
ensure_csv(SCORES_CSV, SCORES_HDR)
ensure_csv(MATCH_STATE_CSV, MATCH_STATE_HDR)

# --------- Match state persistence ----------
def load_state_row(match_id):
    state_df = load_csv(MATCH_STATE_CSV, MATCH_STATE_HDR)
    row = state_df[state_df["match_id"] == match_id]
    if row.empty:
        # initialize a new state record
        new = pd.DataFrame([{
            "match_id": match_id,
            "half": 1,
            "start_time_iso": "",
            "accumulated_seconds": 0.0,
            "is_running": False,
            "team1_timeouts": 0,
            "team2_timeouts": 0
        }])
        state_df = pd.concat([state_df, new], ignore_index=True)
        save_csv(state_df, MATCH_STATE_CSV)
        return new.iloc[0].to_dict()
    else:
        # return as dict
        return row.iloc[0].to_dict()

def save_state_row(state_obj):
    state_df = load_csv(MATCH_STATE_CSV, MATCH_STATE_HDR)
    mid = int(state_obj["match_id"])
    if mid in state_df["match_id"].values:
        idx = state_df[state_df["match_id"] == mid].index[0]
        for k, v in state_obj.items():
            state_df.at[idx, k] = v
    else:
        state_df = pd.concat([state_df, pd.DataFrame([state_obj])], ignore_index=True)
    save_csv(state_df, MATCH_STATE_CSV)

# --------- Helpers ----------
def get_team(row_df, team_id):
    teams = load_csv(TEAMS_CSV, TEAMS_HDR)
    match = teams[teams["team_id"] == int(team_id)]
    if match.empty:
        return {"team_id": team_id, "team_name": f"Team {team_id}"}
    return match.iloc[0].to_dict()

def get_player_name(player_id):
    if player_id == "" or pd.isnull(player_id):
        return ""
    players = load_csv(PLAYERS_CSV, PLAYERS_HDR)
    row = players[players["player_id"] == int(player_id)]
    return row.iloc[0]["player_name"] if not row.empty else str(player_id)

def compute_scores_for_match(match_id):
    scores = load_csv(SCORES_CSV, SCORES_HDR)
    if scores.empty:
        return {}
    m = scores[scores["match_id"] == int(match_id)]
    if m.empty:
        return {}
    totals = m.groupby("team_id")["points"].sum().to_dict()
    # convert keys to int
    new = {}
    for k, v in totals.items():
        try:
            new[int(k)] = int(v)
        except:
            new[k] = int(v)
    return new

# --------- Admin UI ----------
def add_tournament():
    st.subheader("🏆 Add Tournament")
    df = load_csv(TOURNAMENTS_CSV, TOURNAMENTS_HDR)
    name = st.text_input("Tournament Name")
    start = st.date_input("Start Date")
    end = st.date_input("End Date")
    location = st.text_input("Location")
    if st.button("Add Tournament"):
        if not name.strip():
            st.warning("Please enter a name")
            return
        new_id = int(df["tournament_id"].max()) + 1 if not df.empty else 1
        df = pd.concat([df, pd.DataFrame([[new_id, name, str(start), str(end), location]], columns=TOURNAMENTS_HDR)], ignore_index=True)
        save_csv(df, TOURNAMENTS_CSV)
        st.success(f"Tournament '{name}' added (ID {new_id})")

def add_team():
    st.subheader("➕ Add Team")
    teams = load_csv(TEAMS_CSV, TEAMS_HDR)
    tours = load_csv(TOURNAMENTS_CSV, TOURNAMENTS_HDR)
    if tours.empty:
        st.warning("Add a tournament first.")
        return
    tour_choice = st.selectbox("Tournament", tours["tournament_name"].tolist())
    team_name = st.text_input("Team Name")
    if st.button("Add Team"):
        if not team_name.strip():
            st.warning("Enter team name")
            return
        tournament_id = int(tours[tours["tournament_name"] == tour_choice]["tournament_id"].iloc[0])
        new_id = int(teams["team_id"].max()) + 1 if not teams.empty else 1
        teams = pd.concat([teams, pd.DataFrame([[new_id, team_name, tournament_id]], columns=TEAMS_HDR)], ignore_index=True)
        save_csv(teams, TEAMS_CSV)
        st.success(f"Team '{team_name}' added (ID {new_id})")

def add_player():
    st.subheader("➕ Add Player")
    players = load_csv(PLAYERS_CSV, PLAYERS_HDR)
    teams = load_csv(TEAMS_CSV, TEAMS_HDR)
    if teams.empty:
        st.warning("Add teams first.")
        return
    team_choice = st.selectbox("Team", teams["team_name"].tolist())
    player_name = st.text_input("Player Name")
    if st.button("Add Player"):
        if not player_name.strip():
            st.warning("Enter player name")
            return
        team_id = int(teams[teams["team_name"] == team_choice]["team_id"].iloc[0])
        new_id = int(players["player_id"].max()) + 1 if not players.empty else 1
        players = pd.concat([players, pd.DataFrame([[new_id, player_name, team_id]], columns=PLAYERS_HDR)], ignore_index=True)
        save_csv(players, PLAYERS_CSV)
        st.success(f"Player '{player_name}' added (ID {new_id})")

def schedule_match():
    st.subheader("📅 Schedule Match")
    matches = load_csv(MATCHES_CSV, MATCHES_HDR)
    teams = load_csv(TEAMS_CSV, TEAMS_HDR)
    tours = load_csv(TOURNAMENTS_CSV, TOURNAMENTS_HDR)
    if tours.empty or teams.empty:
        st.warning("Add tournaments and teams first.")
        return
    tour_choice = st.selectbox("Tournament", tours["tournament_name"].tolist())
    tournament_id = int(tours[tours["tournament_name"] == tour_choice]["tournament_id"].iloc[0])
    eligible = teams[teams["tournament_id"] == tournament_id]
    if eligible.shape[0] < 2:
        st.warning("Need at least 2 teams in the tournament.")
        return
    team1 = st.selectbox("Team 1", eligible["team_name"].tolist())
    team2 = st.selectbox("Team 2", [t for t in eligible["team_name"].tolist() if t != team1])
    match_date = st.date_input("Match Date")
    venue = st.text_input("Venue")
    if st.button("Schedule Match"):
        new_id = int(matches["match_id"].max()) + 1 if not matches.empty else 1
        t1_id = int(eligible[eligible["team_name"] == team1]["team_id"].iloc[0])
        t2_id = int(eligible[eligible["team_name"] == team2]["team_id"].iloc[0])
        matches = pd.concat([matches, pd.DataFrame([[new_id, tournament_id, t1_id, t2_id, str(match_date), venue]], columns=MATCHES_HDR)], ignore_index=True)
        save_csv(matches, MATCHES_CSV)
        st.success(f"Match scheduled (ID {new_id})")

# --------- Live match UI ----------
def render_scoreboard(team1_row, team2_row, team1_points, team2_points):
    col1, colmid, col2 = st.columns([3, 1, 3])
    with col1:
        st.markdown(f"### {team1_row['team_name']}")
        st.markdown(f"## {team1_points}")
    with colmid:
        st.markdown("### VS")
    with col2:
        st.markdown(f"### {team2_row['team_name']}")
        st.markdown(f"## {team2_points}")

def load_state_and_compute(match_id):
    state = load_state_row(int(match_id))
    # compute elapsed (accumulated + running)
    accumulated = float(state["accumulated_seconds"]) if state["accumulated_seconds"] != "" else 0.0
    if state["is_running"] and state["start_time_iso"]:
        try:
            start_dt = datetime.fromisoformat(state["start_time_iso"])
            delta = (datetime.utcnow() - start_dt).total_seconds()
            accumulated += delta
        except Exception:
            # malformed iso string — ignore and treat as paused
            pass
    # determine half automatically if necessary (but we persist half)
    # switch half if accumulated >= HALF_SECONDS and half == 1
    if accumulated >= HALF_SECONDS and int(state["half"]) == 1:
        state["half"] = 2
        # set accumulated_seconds to value >= HALF_SECONDS so 2nd half elapsed starts from HALF_SECONDS
        state["accumulated_seconds"] = float(accumulated)
        save_state_row(state)
    # compute elapsed_in_half and remaining
    total_elapsed = accumulated
    if int(state["half"]) == 1:
        elapsed_in_half = min(total_elapsed, HALF_SECONDS)
        remaining_in_half = max(HALF_SECONDS - elapsed_in_half, 0)
    else:
        elapsed_in_half = max(0, min(total_elapsed - HALF_SECONDS, HALF_SECONDS))
        remaining_in_half = max(HALF_SECONDS - elapsed_in_half, 0)
    # auto end match if total_elapsed >= TOTAL_SECONDS
    match_over = total_elapsed >= TOTAL_SECONDS
    return state, total_elapsed, elapsed_in_half, remaining_in_half, match_over

def update_live_match():
    st.subheader("🔴 Live Match")
    matches = load_csv(MATCHES_CSV, MATCHES_HDR)
    teams = load_csv(TEAMS_CSV, TEAMS_HDR)
    players = load_csv(PLAYERS_CSV, PLAYERS_HDR)
    scores = load_csv(SCORES_CSV, SCORES_HDR)

    if matches.empty:
        st.warning("No matches scheduled.")
        return

    # select match
    teams_df = load_csv(TEAMS_CSV)
    match_options = [
    f"{int(r['match_id'])} | {r['match_date']} | {get_team(teams_df, int(r['team1_id']))['team_name']} vs {get_team(teams_df, int(r['team2_id']))['team_name']}"
    for _, r in matches.iterrows()
]
    sel = st.selectbox("Select Match", match_options)
    mid = int(sel.split("|")[0].strip())
    match_row = matches[matches["match_id"] == mid].iloc[0]
    teams_df = load_csv(TEAMS_CSV)
    team1_row = get_team(teams_df, match_row["team1_id"])
    team2_row = get_team(teams_df, match_row["team2_id"])

    # compute totals
    totals = compute_scores_for_match(mid)
    t1_points = totals.get(int(team1_row["team_id"]), 0)
    t2_points = totals.get(int(team2_row["team_id"]), 0)

    render_scoreboard(team1_row, team2_row, t1_points, t2_points)

    # controls
    st.markdown("### ⏱ Clock Controls")
    cols = st.columns(4)
    start_btn = cols[0].button("▶ Start / Resume")
    pause_btn = cols[1].button("⏸ Pause")
    reset_btn = cols[2].button("🔄 Reset Match")
    autoupdate = cols[3].checkbox("Auto refresh every 1s (blocks UI)", value=False)

    # load state
    state = load_state_row(mid)

    # start/resume
    if start_btn:
        if not state["is_running"]:
            state["is_running"] = True
            state["start_time_iso"] = datetime.utcnow().isoformat()
            save_state_row(state)
            st.success("Timer started/resumed.")
        else:
            st.info("Timer already running.")

    # pause
    if pause_btn:
        if state["is_running"] and state["start_time_iso"]:
            try:
                start_dt = datetime.fromisoformat(state["start_time_iso"])
                delta = (datetime.utcnow() - start_dt).total_seconds()
            except Exception:
                delta = 0.0
            state["accumulated_seconds"] = float(state["accumulated_seconds"]) + delta
            state["is_running"] = False
            state["start_time_iso"] = ""
            save_state_row(state)
            st.success("Timer paused.")
        else:
            st.info("Timer not running.")

    # reset
    if reset_btn:
        state = {"match_id": mid, "half": 1, "start_time_iso": "", "accumulated_seconds": 0.0, "is_running": False, "team1_timeouts": 0, "team2_timeouts": 0}
        save_state_row(state)
        st.success("Match reset.")

    # compute updated timing info
    state, total_elapsed, elapsed_in_half, remaining_in_half, match_over = load_state_and_compute(mid)

    # auto-switch half if required (handled in load_state_and_compute); persist again just in case
    save_state_row(state)

    # display times
    st.markdown(f"**Current Half:** {int(state['half'])}")
    st.markdown(f"**Elapsed this half:** {int(elapsed_in_half)//60:02d}:{int(elapsed_in_half)%60:02d}")
    st.markdown(f"**Remaining this half:** {int(remaining_in_half)//60:02d}:{int(remaining_in_half)%60:02d}")
    st.markdown(f"**Total elapsed:** {int(total_elapsed)//60:02d}:{int(total_elapsed)%60:02d}")

    if match_over:
        st.success("🏁 Match ended (40 minutes). Timer stopped.")
        # ensure stopped
        if state["is_running"]:
            state["is_running"] = False
            state["start_time_iso"] = ""
            save_state_row(state)

    # Auto-refresh (simple loop updating a placeholder)
    if autoupdate and not match_over:
        placeholder = st.empty()
        # block UI for up to 60 seconds or until match ends
        for i in range(60):
            time.sleep(1)
            state, total_elapsed, elapsed_in_half, remaining_in_half, match_over = load_state_and_compute(mid)
            # update scoreboard & time in the placeholder
            with placeholder.container():
                # recompute totals
                totals = compute_scores_for_match(mid)
                t1_points = totals.get(int(team1_row["team_id"]), 0)
                t2_points = totals.get(int(team2_row["team_id"]), 0)
                render_scoreboard(team1_row, team2_row, t1_points, t2_points)
                st.markdown(f"**Current Half:** {int(state['half'])}")
                st.markdown(f"**Elapsed this half:** {int(elapsed_in_half)//60:02d}:{int(elapsed_in_half)%60:02d}")
                st.markdown(f"**Remaining this half:** {int(remaining_in_half)//60:02d}:{int(remaining_in_half)%60:02d}")
                st.markdown(f"**Total elapsed:** {int(total_elapsed)//60:02d}:{int(total_elapsed)%60:02d}")
            if match_over:
                break
        placeholder.empty()

    # --- Event Recording ---
    st.markdown("---")
    st.markdown("### ➕ Record Event")
    event_team_name = st.selectbox("Team", [team1_row["team_name"], team2_row["team_name"]])
    team_id_val = int(team1_row["team_id"]) if event_team_name == team1_row["team_name"] else int(team2_row["team_id"])
    # prepare player list for that team
    players_df = load_csv(PLAYERS_CSV, PLAYERS_HDR)
    team_players = players_df[players_df["team_id"] == team_id_val]["player_name"].tolist()
    player_choice = st.selectbox("Player (if substituting you can use free text)", team_players + ["(Unknown / Free-text)"])
    if player_choice == "(Unknown / Free-text)":
        player_name_free = st.text_input("Player name (free-text)")
        # try to find if exists already else leave player_id blank
        pid = None
        if player_name_free.strip():
            matchp = players_df[players_df["player_name"] == player_name_free.strip()]
            if not matchp.empty:
                pid = int(matchp["player_id"].iloc[0])
            else:
                pid = ""
    else:
        # find player id
        matchp = players_df[players_df["player_name"] == player_choice]
        pid = int(matchp["player_id"].iloc[0]) if not matchp.empty else ""

    event_type = st.selectbox("Event Type", ["Raid", "Tackle", "Bonus", "Foul", "Timeout", "Substitution"])
    points = 0
    details = ""
    if event_type in ["Raid", "Tackle", "Bonus"]:
        points = st.number_input("Points awarded", min_value=0, max_value=5, value=1)
    elif event_type == "Foul":
        details = st.text_input("Details about the foul (optional)")
    elif event_type == "Timeout":
        # show how many used
        used = int(state["team1_timeouts"]) if team_id_val == int(team1_row["team_id"]) else int(state["team2_timeouts"])
        st.markdown(f"Timeouts used by {event_team_name}: {used}/{MAX_TIMEOUTS_PER_TEAM}")
    elif event_type == "Substitution":
        sub_in = st.text_input("Substitute IN (player name)")
        sub_out = st.selectbox("Substitute OUT", team_players if len(team_players) > 0 else ["(none)"])
        details = f"sub_in:{sub_in};sub_out:{sub_out}"

    # minute default
    current_minute = int(total_elapsed // 60)
    minute = st.number_input("Minute (defaults to current minute)", min_value=0, max_value=40, value=current_minute)

    if st.button("Record Event"):
        # enforce timeout count limit
        if event_type == "Timeout":
            if team_id_val == int(team1_row["team_id"]):
                if int(state["team1_timeouts"]) >= MAX_TIMEOUTS_PER_TEAM:
                    st.error(f"{team1_row['team_name']} has used all timeouts.")
                    return
                else:
                    state["team1_timeouts"] = int(state["team1_timeouts"]) + 1
            else:
                if int(state["team2_timeouts"]) >= MAX_TIMEOUTS_PER_TEAM:
                    st.error(f"{team2_row['team_name']} has used all timeouts.")
                    return
                else:
                    state["team2_timeouts"] = int(state["team2_timeouts"]) + 1
            save_state_row(state)
            points_val = 0
        elif event_type == "Substitution":
            points_val = 0
        else:
            points_val = int(points)

        # build row
        new_row = {
            "match_id": int(mid),
            "minute": int(minute),
            "event_type": event_type,
            "player_id": int(pid) if pid not in (None, "", "(none)") else "",
            "team_id": int(team_id_val),
            "points": int(points_val),
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        scores_df = load_csv(SCORES_CSV, SCORES_HDR)
        scores_df = pd.concat([scores_df, pd.DataFrame([new_row])], ignore_index=True)
        save_csv(scores_df, SCORES_CSV)
        st.success("Event recorded.")

    # Undo last event for this match
    st.markdown("### ↩ Undo Last Event (this match)")
    scores_df = load_csv(SCORES_CSV, SCORES_HDR)
    match_scores = scores_df[scores_df["match_id"] == int(mid)]
    if not match_scores.empty:
        last_idx = match_scores.tail(1).index[-1]
        st.write(match_scores.tail(1)[["minute", "event_type", "player_id", "team_id", "points", "details", "timestamp"]])
        if st.button("Undo Last for this match"):
            scores_df = scores_df.drop(index=last_idx)
            save_csv(scores_df, SCORES_CSV)
            st.success("Last event removed.")
    else:
        st.info("No events yet for this match.")

    # Recent events table
    st.markdown("### Recent events (latest first)")
    scores_df = load_csv(SCORES_CSV, SCORES_HDR)
    if not scores_df.empty:
        display = scores_df[scores_df["match_id"] == int(mid)].sort_values("timestamp", ascending=False).head(20)
        # merge player/team names
        players_df = load_csv(PLAYERS_CSV, PLAYERS_HDR)
        teams_df = load_csv(TEAMS_CSV, TEAMS_HDR)
        if not display.empty:
            display = display.merge(players_df[["player_id", "player_name"]], how="left", on="player_id")
            display = display.merge(teams_df[["team_id", "team_name"]], how="left", on="team_id")
            display = display[["minute", "event_type", "player_name", "team_name", "points", "details", "timestamp"]]
            st.dataframe(display)
    else:
        st.info("No events recorded yet.")

# --------- Summary view ----------
def view_summary():
    st.subheader("📊 Match Summary")
    scores_df = load_csv(SCORES_CSV, SCORES_HDR)
    if scores_df.empty:
        st.warning("No events recorded.")
        return
    matches = load_csv(MATCHES_CSV, MATCHES_HDR)
    match_choice = st.selectbox("Select match", matches["match_id"].tolist())
    m = int(match_choice)
    match_scores = scores_df[scores_df["match_id"] == m]
    if match_scores.empty:
        st.info("No events for this match.")
        return
    teams = load_csv(TEAMS_CSV, TEAMS_HDR)
    players = load_csv(PLAYERS_CSV, PLAYERS_HDR)
    totals = match_scores.groupby("team_id")["points"].sum().reset_index()
    totals = totals.merge(teams[["team_id", "team_name"]], on="team_id", how="left")
    st.table(totals)
    player_totals = match_scores.groupby("player_id")["points"].sum().reset_index()
    player_totals = player_totals.merge(players[["player_id", "player_name"]], on="player_id", how="left")
    st.table(player_totals.sort_values("points", ascending=False))
def run():
    st.title("Kabaddi Dashboard")
    st.write("Live kabaddi stats go here...")
# --------- Main runner ----------
def run_kabaddi():
    st.sidebar.title("kabaddi Module")
    menu = ["Add Tournament", "Add Team", "Add Player", "View Tournaments", "Schedule Match", "Update Live Match", "View Match Summary"]
    choice = st.sidebar.radio("Menu", menu)

    if choice == "Add Tournament":
        add_tournament()
    elif choice == "Add Team":
        add_team()
    elif choice == "Add Player":
        add_player()
    elif choice == "View Tournaments":
        st.subheader("Tournaments")
        df = load_csv(TOURNAMENTS_CSV, TOURNAMENTS_HDR)
        st.dataframe(df)
    elif choice == "Schedule Match":
        schedule_match()
    elif choice == "Update Live Match":
        update_live_match()
    elif choice == "View Match Summary":
        view_summary()



