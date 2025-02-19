import streamlit as st
import pandas as pd
import requests
from collections import Counter

# Constants
BUDGET = 100.0
TEAM_LIMIT = 3
SQUAD_SIZE = 15
STARTING_11_SIZE = 11
POSITION_LIMITS = {
    'Goalkeeper': 2,
    'Defender': 5,
    'Midfielder': 5,
    'Forward': 3
}
STARTING_MINIMUMS = {
    'Goalkeeper': 1,
    'Defender': 3,
    'Midfielder': 3,
    'Forward': 1
}
POSITION_MAP = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}

# Fetch FPL Data
@st.cache_data
def fetch_fpl_data():
    players_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"

    try:
        players_data = requests.get(players_url).json()
        fixtures_data = requests.get(fixtures_url).json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame(), pd.DataFrame()

    players_df = pd.DataFrame(players_data['elements'])
    teams_df = pd.DataFrame(players_data['teams'])
    positions_df = pd.DataFrame(players_data['element_types'])
    
    team_map = teams_df.set_index('id')['name'].to_dict()
    players_df['team_name'] = players_df['team'].map(team_map)
    players_df['position'] = players_df['element_type'].map(POSITION_MAP)
    players_df['now_cost'] = players_df['now_cost'] / 10
    players_df['expected_points'] = players_df['ep_next'].astype(float)
    
    fixtures_df = pd.DataFrame(fixtures_data)
    fixtures_df['team_h'] = fixtures_df['team_h'].map(team_map)
    fixtures_df['team_a'] = fixtures_df['team_a'].map(team_map)
    
    return players_df, fixtures_df

# Get next 3 fixtures for each player
def get_next_fixtures(team_name, fixtures_df):
    future_fixtures = fixtures_df[(fixtures_df['team_h'] == team_name) | (fixtures_df['team_a'] == team_name)]
    next_3 = future_fixtures.head(3)
    fixtures_list = [f"{row['team_h']} vs {row['team_a']}" for _, row in next_3.iterrows()]
    return ', '.join(fixtures_list)

# Team Selection
def select_starting_11(players_df, selected_ids, budget_left, team_counts, position_counts):
    starters = []
    starters_sorted = players_df.sort_values(by='expected_points', ascending=False)
    
    for _, player in starters_sorted.iterrows():
        pos = player['position']
        team_name = player['team_name']
        price = player['now_cost']
        player_id = player['id']

        if (player_id not in selected_ids and
            position_counts[pos] < POSITION_LIMITS[pos] and
            team_counts[team_name] < TEAM_LIMIT and
            budget_left - price >= 0):
            starters.append(player)
            selected_ids.add(player_id)
            position_counts[pos] += 1
            team_counts[team_name] += 1
            budget_left -= price

            if len(starters) == STARTING_11_SIZE:
                break

    return pd.DataFrame(starters), selected_ids, budget_left

def select_bench(players_df, selected_ids, budget_left, team_counts, position_counts):
    bench = []
    bench_sorted = players_df[~players_df['id'].isin(selected_ids)].sort_values(by=['now_cost', 'expected_points'], ascending=[True, False])
    for _, player in bench_sorted.iterrows():
        pos = player['position']
        team_name = player['team_name']
        price = player['now_cost']
        player_id = player['id']

        if (player_id not in selected_ids and
            position_counts[pos] < POSITION_LIMITS[pos] and
            team_counts[team_name] < TEAM_LIMIT and
            budget_left - price >= 0):
            bench.append(player)
            selected_ids.add(player_id)
            position_counts[pos] += 1
            team_counts[team_name] += 1
            budget_left -= price

            if len(bench) == SQUAD_SIZE - STARTING_11_SIZE:
                break
    return pd.DataFrame(bench), budget_left

def upgrade_bench(bench_df, players_df, selected_ids, budget_left, team_counts):
    upgrades_log = []
    bench_df = bench_df.reset_index(drop=True)
    
    for i in range(len(bench_df)):
        bench_player = bench_df.iloc[i]
        upgrades = players_df[
            (~players_df['id'].isin(selected_ids)) &
            (players_df['position'] == bench_player['position']) &
            (players_df['now_cost'] <= bench_player['now_cost'] + budget_left)
        ]
        upgrades = upgrades[upgrades['team_name'].apply(lambda x: team_counts[x] < TEAM_LIMIT)]
        upgrades = upgrades.sort_values(by='expected_points', ascending=False)

        if not upgrades.empty:
            upgrade = upgrades.iloc[0]
            budget_left += bench_player['now_cost'] - upgrade['now_cost']
            team_counts[bench_player['team_name']] -= 1
            team_counts[upgrade['team_name']] += 1
            bench_df.iloc[i] = upgrade
            selected_ids.remove(bench_player['id'])
            selected_ids.add(upgrade['id'])
            upgrades_log.append(f"Upgraded {bench_player['web_name']} → {upgrade['web_name']}")
    
    return bench_df, upgrades_log

def main():
    st.title("FPL Team Selection")
    players_df, fixtures_df = fetch_fpl_data()
    players_df['next_fixtures'] = players_df['team_name'].apply(lambda x: get_next_fixtures(x, fixtures_df))

    if st.button("Generate Optimal Team"):
        selected_ids = set()
        team_counts = Counter()
        position_counts = Counter()
        budget_left = BUDGET

        starting_11_df, selected_ids, budget_left = select_starting_11(players_df, selected_ids, budget_left, team_counts, position_counts)
        bench_df, budget_left = select_bench(players_df, selected_ids, budget_left, team_counts, position_counts)
        bench_df, upgrades_log = upgrade_bench(bench_df, players_df, selected_ids, budget_left, team_counts)

        total_ep_starting_11 = starting_11_df['expected_points'].sum()
        total_budget_used = starting_11_df['now_cost'].sum() + bench_df['now_cost'].sum()

        st.subheader(f"Selected Starting 11 (Total Expected Points: {total_ep_starting_11:.2f})")
        st.dataframe(starting_11_df[['web_name', 'team_name', 'position', 'now_cost', 'expected_points', 'next_fixtures']])

        st.subheader("Bench")
        st.dataframe(bench_df[['web_name', 'team_name', 'position', 'now_cost', 'expected_points', 'next_fixtures']])

        st.subheader(f"Total Budget Used: £{total_budget_used:.1f}m")
        if upgrades_log:
            st.subheader("Upgrades")
            st.text("\n".join(upgrades_log))

if __name__ == "__main__":
    main()