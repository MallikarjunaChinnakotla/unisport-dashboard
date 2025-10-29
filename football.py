import streamlit as st
from sports.background import set_bg
import pandas as pd
import os
import time
from datetime import datetime

# ---------- Paths ----------
DATA_PATH = "folder_path"
TOURNAMENTS_CSV = os.path.join(DATA_PATH, "football_tournaments.csv")
TEAMS_CSV = os.path.join(DATA_PATH, "football_teams.csv")
PLAYERS_CSV = os.path.join(DATA_PATH, "football_players.csv")
MATCHES_CSV = os.path.join(DATA_PATH, "football_matches.csv")
SCORES_CSV = os.path.join(DATA_PATH, "football_scores.csv")

MATCH_PHASES = ["1st Half", "Half-Time", "2nd Half", "Full-Time"]



# ---------- Utils ----------
def load_csv(path, columns=None):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()

def save_csv(df, path):
    os.makedirs(DATA_PATH, exist_ok=True)
    df.to_csv(path, index=False)

# ---------- Add Tournament ----------
def add_tournament():
    st.subheader("🏆 Add Football Tournament")
    tournaments = load_csv(TOURNAMENTS_CSV, 
                           ["tournament_id", "tournament_name", "start_date", "end_date", "location"])

    tournament_name = st.text_input("Tournament Name")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    location = st.text_input("Location")

    if st.button("Add Tournament") and tournament_name:
        tournament_id = tournaments["tournament_id"].max() + 1 if not tournaments.empty else 1
        new_row = pd.DataFrame([[tournament_id, tournament_name, start_date, end_date, location]],
                               columns=["tournament_id", "tournament_name", "start_date", "end_date", "location"])
        tournaments = pd.concat([tournaments, new_row], ignore_index=True)
        save_csv(tournaments, TOURNAMENTS_CSV)
        st.success(f"Tournament '{tournament_name}' added successfully!")

# ---------- View Tournaments ----------
def view_tournaments():
    st.subheader("📋 View Tournaments")
    tournaments = load_csv(TOURNAMENTS_CSV)
    matches = load_csv(MATCHES_CSV)
    teams = load_csv(TEAMS_CSV)

    if tournaments.empty:
        st.warning("No tournaments found.")
        return

    tournament_names = tournaments["tournament_name"].tolist()
    selected_tournament = st.selectbox("Select Tournament", tournament_names)

    if selected_tournament:
        st.markdown(f"### Matches in {selected_tournament}")
        tournament_id = tournaments.loc[tournaments["tournament_name"] == selected_tournament, "tournament_id"].values[0]
        tournament_matches = matches[matches["tournament_id"] == tournament_id]

        if tournament_matches.empty:
            st.info("No matches scheduled for this tournament yet.")
        else:
            def get_team_name(team_id):
                name = teams.loc[teams["team_id"] == team_id, "team_name"]
                return name.values[0] if not name.empty else "Unknown"

            display_df = tournament_matches.copy()
            display_df["Team 1"] = display_df["team1_id"].apply(get_team_name)
            display_df["Team 2"] = display_df["team2_id"].apply(get_team_name)
            display_df["Match Date"] = display_df["match_date"]
            display_df["Venue"] = display_df["venue"]

            st.dataframe(display_df[["match_id", "Team 1", "Team 2", "Match Date", "Venue"]].reset_index(drop=True))

# ---------- Add Team ----------
def add_team():
    st.subheader("⚽ Add Football Team")
    tournaments = load_csv(TOURNAMENTS_CSV, ["tournament_id", "tournament_name", "start_date", "end_date", "location"])
    if tournaments.empty:
        st.warning("No tournaments found. Please add a tournament first.")
        return

    tournament_choice = st.selectbox("Select Tournament", tournaments["tournament_name"])
    tournament_id = tournaments.loc[tournaments["tournament_name"] == tournament_choice, "tournament_id"].values[0]

    team_name = st.text_input("Team Name")

    if st.button("Add Team") and team_name:
        teams = load_csv(TEAMS_CSV, ["team_id", "team_name", "tournament_id"])
        team_id = teams["team_id"].max() + 1 if not teams.empty else 1
        new_row = pd.DataFrame([[team_id, team_name, tournament_id]], columns=teams.columns)
        teams = pd.concat([teams, new_row], ignore_index=True)
        save_csv(teams, TEAMS_CSV)
        st.success(f"Team '{team_name}' added successfully!")

# ---------- Add Player ----------
def add_player():
    st.subheader("👤 Add Football Player")
    teams = load_csv(TEAMS_CSV, ["team_id", "team_name", "tournament_id"])
    if teams.empty:
        st.warning("No teams found. Please add a team first.")
        return

    team_choice = st.selectbox("Select Team", teams["team_name"])
    team_id = teams.loc[teams["team_name"] == team_choice, "team_id"].values[0]
    player_name = st.text_input("Player Name")

    if st.button("Add Player"):
        if not player_name.strip():
            st.warning("Please enter a player name.")
            return

        df = load_csv(PLAYERS_CSV, ["player_id", "player_name", "team_id"])

        player_id = df["player_id"].max() + 1 if not df.empty else 1

        new_row = pd.DataFrame([[player_id, player_name.strip(), team_id]], 
                               columns=["player_id", "player_name", "team_id"])
        df = pd.concat([df, new_row], ignore_index=True)
        save_csv(df, PLAYERS_CSV)
        st.success(f"Player '{player_name}' added successfully!")

# ---------- Schedule Match ----------
def schedule_match():
    st.subheader("📅 Schedule Match")
    tournaments = load_csv(TOURNAMENTS_CSV)
    teams = load_csv(TEAMS_CSV)

    if tournaments.empty or teams.empty:
        st.warning("Add tournaments and teams first.")
        return

    t_name = st.selectbox("Tournament", tournaments["tournament_name"])
    tid = tournaments[tournaments["tournament_name"] == t_name]["tournament_id"].iloc[0]
    team_list = teams[teams["tournament_id"] == tid]["team_name"].tolist()

    t1 = st.selectbox("Team 1", team_list)
    t2 = st.selectbox("Team 2", [x for x in team_list if x != t1])
    match_date = st.date_input("Match Date")
    venue = st.text_input("Venue")

    if st.button("Schedule Match"):
        df = load_csv(MATCHES_CSV, ["match_id", "tournament_id", "team1_id", "team2_id", "match_date", "venue"])
        mid = df["match_id"].max() + 1 if not df.empty else 1
        team1_id = teams[teams["team_name"] == t1]["team_id"].iloc[0]
        team2_id = teams[teams["team_name"] == t2]["team_id"].iloc[0]

        df = pd.concat([df, pd.DataFrame([[mid, tid, team1_id, team2_id, match_date, venue]],
                                         columns=["match_id", "tournament_id", "team1_id", "team2_id", "match_date", "venue"])],
                       ignore_index=True)
        save_csv(df, MATCHES_CSV)
        st.success(f"Match scheduled with ID {mid}")

def update_score():
    st.subheader("⚽ Live Match Events")

    matches = load_csv(MATCHES_CSV)
    teams = load_csv(TEAMS_CSV)
    players = load_csv(PLAYERS_CSV)
    scores = load_csv(SCORES_CSV, ["match_id", "minute", "event_type", "team_name", "player_name", "timestamp"])

    if matches.empty:
        st.warning("No matches available.")
        return

    # Select match
    match_id = st.selectbox("Select Match ID", matches["match_id"].unique())
    match = matches[matches["match_id"] == match_id].iloc[0]

    team1_id = match["team1_id"]
    team2_id = match["team2_id"]

    team1_name = teams.loc[teams["team_id"] == team1_id, "team_name"].values[0]
    team2_name = teams.loc[teams["team_id"] == team2_id, "team_name"].values[0]

    # Get players per team
    players_team1 = players[players["team_id"] == team1_id]["player_name"].tolist()
    players_team2 = players[players["team_id"] == team2_id]["player_name"].tolist()

    # -------- TIMER SETUP --------
    total_duration_seconds = 90 * 60  # 90 minutes

    # Initialize session state vars
    if "timer_start" not in st.session_state or st.session_state.get("current_match_id") != match_id:
        st.session_state["timer_start"] = time.time()
        st.session_state["timer_paused"] = False
        st.session_state["paused_time"] = 0
        st.session_state["pause_start"] = None
        st.session_state["current_match_id"] = match_id

    # Pause / Resume controls
    col1, col2 = st.columns(2)
    if col1.button("Pause Timer"):
        if not st.session_state.timer_paused:
            st.session_state.pause_start = time.time()
            st.session_state.timer_paused = True
    if col2.button("Resume Timer"):
        if st.session_state.timer_paused:
            st.session_state.paused_time += time.time() - st.session_state.pause_start
            st.session_state.pause_start = None
            st.session_state.timer_paused = False

    # Calculate elapsed and remaining time
    if st.session_state.timer_paused:
        elapsed_seconds = st.session_state.pause_start - st.session_state.timer_start - st.session_state.paused_time
    else:
        elapsed_seconds = time.time() - st.session_state.timer_start - st.session_state.paused_time

    remaining_seconds = max(total_duration_seconds - elapsed_seconds, 0)
    remaining_minutes = int(remaining_seconds // 60)
    remaining_secs = int(remaining_seconds % 60)

    st.markdown(f"⏳ Match Timer: **{remaining_minutes:02d}:{remaining_secs:02d}** remaining")

    if remaining_seconds == 0:
        st.warning("⏰ Match time is over. No more events can be added.")
        # Optionally you can disable inputs here or just return
        return

    # Trigger auto rerun every second only if not paused and time remains
    if not st.session_state.timer_paused and remaining_seconds > 0:
        time.sleep(1)
        

    # -------- INPUTS FOR GOAL --------
    scoring_team = st.selectbox("Select Scoring Team", [team1_name, team2_name])

    scorer_list = players_team1 if scoring_team == team1_name else players_team2

    if scorer_list:
        scorer = st.selectbox("Select Goal Scorer", scorer_list)
    else:
        st.warning(f"No players found for {scoring_team}.")
        return

    # Minute of goal input defaults to elapsed minutes capped at 90
    default_minute = min(int(elapsed_seconds // 60), 90)
    minute_of_goal = st.number_input(
        "Minute of Goal",
        min_value=0,
        max_value=90,
        value=default_minute,
        step=1
    )

    if st.button("Add Goal"):
        new_event = [match_id, minute_of_goal, "Goal", scoring_team, scorer, datetime.now()]
        new_row = pd.DataFrame([new_event], columns=["match_id", "minute", "event_type", "team_name", "player_name", "timestamp"])
        scores = pd.concat([scores, new_row], ignore_index=True)
        save_csv(scores, SCORES_CSV)
        st.success(f"Goal recorded: {scorer} for {scoring_team} at minute {minute_of_goal}!")

    # Show current score
    if not scores.empty:
        match_scores = scores[(scores["match_id"] == match_id) & (scores["event_type"] == "Goal")]
        tally = match_scores.groupby("team_name").size().to_dict()
        team1_score = tally.get(team1_name, 0)
        team2_score = tally.get(team2_name, 0)
        st.markdown(f"### Current Score: **{team1_name} {team1_score} - {team2_score} {team2_name}**")


# ---------- View Match Summary ----------
def view_summary():
    st.subheader("📊 Match Summary")
    match_id = st.number_input("Enter Match ID", min_value=1, step=1)
    df = load_csv(SCORES_CSV)

    if not df.empty:
        match_df = df[df["match_id"] == match_id]
        if match_df.empty:
            st.warning("No events found for this match.")
        else:
            st.dataframe(match_df)
    else:
        st.warning("No events found for this match.")
def run():
    st.title("⚽ Football  Dashboard")
    st.write("Live Football  stats go here...")
# ---------- Main ----------
def run_football():
    st.sidebar.title("⚽ Football Module")
    choice = st.sidebar.radio("Select Option", [
        "Add Tournament", "Add Team", "Add Player", "View Tournaments", "Schedule Match", "Update Live Score", "View Match Summary"
    ])

    if choice == "Add Tournament":
        add_tournament()
    elif choice == "Add Team":
        add_team()
    elif choice == "Add Player":
        add_player()
    elif choice == "View Tournaments":
        view_tournaments()
    elif choice == "Schedule Match":
        schedule_match()
    elif choice == "Update Live Score":
        update_score()
    elif choice == "View Match Summary":
        view_summary()


