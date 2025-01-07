import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time

load_dotenv()  # Load .env file

email = os.getenv("FPL_EMAIL")
password = os.getenv("FPL_PASSWORD")

class FPLAgent(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_url = "https://fantasy.premierleague.com/api/"  # Shared base URL

    def fetch_data(self, endpoint: str) -> Dict:
        """
        Fetch data from the FPL API for the given endpoint.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching data from {url}: {e}")
            return {}

    @abstractmethod
    async def process(self, data: Dict) -> Dict:
        """Process data and return results"""
        pass

class DataScraperAgent(FPLAgent):
    def __init__(self, email: str, password: str):
        super().__init__()
        self.email = email
        self.password = password
        self.session = requests.Session()  # Retain for subsequent API calls

    def authenticate_with_selenium(self):
        """
        Authenticate with the FPL website using Selenium.
        """
        # Configure Selenium WebDriver

        chrome_options = Options()
        # chrome_options.add_argument("--headless")  # Run without a UI
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_service = Service("/opt/homebrew/bin/chromedriver")  # Adjust the path to chromedriver

        driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
        try:
            login_url = "https://users.premierleague.com/accounts/login/"
            driver.get(login_url)
            time.sleep(2)  # Allow time for the page to load
            
            # Find and fill in the email and password fields
            email_field = driver.find_element(By.ID, "ism-email")  # Update ID if necessary
            password_field = driver.find_element(By.ID, "ism-password")  # Update ID if necessary
            email_field.send_keys(self.email)
            password_field.send_keys(self.password)
            
            # Submit the form
            password_field.send_keys(Keys.RETURN)
            time.sleep(5)  # Wait for authentication to complete
            
            # Extract cookies from Selenium
            cookies = driver.get_cookies()
            for cookie in cookies:
                self.session.cookies.set(cookie['name'], cookie['value'])
            
            # Verify authentication
            csrftoken = self.session.cookies.get("csrftoken")
            if not csrftoken:
                raise Exception("Failed to retrieve CSRF token after login.")
            self.logger.info("Successfully authenticated using Selenium.")
        finally:
            driver.quit()

    def fetch_user_team(self):
        """
        Fetch the user's current FPL team.
        """
        user_team_url = "https://fantasy.premierleague.com/api/my-team/"
        response = self.session.get(user_team_url)
        if response.status_code != 200:
            raise Exception("Failed to fetch team data.")
        return response.json()

    async def process(self, data: Dict) -> Dict:
        """Scrape and aggregate FPL data"""
        try:
            # Fetch general data
            general_info = self.fetch_data("bootstrap-static/")
            fixtures = self.fetch_data("fixtures/")
            
            # Process player data
            players_df = pd.DataFrame(general_info['elements'])
            teams_df = pd.DataFrame(general_info['teams'])
            fixtures_df = pd.DataFrame(fixtures)
            
            # Add team name to players
            players_df['team_name'] = players_df['team'].map(
                teams_df.set_index('id')['name']
            )
            
            # Process fixtures
            if not fixtures_df.empty:
                fixtures_df['team_h_name'] = fixtures_df['team_h'].map(
                    teams_df.set_index('id')['name']
                )
                fixtures_df['team_a_name'] = fixtures_df['team_a'].map(
                    teams_df.set_index('id')['name']
                )
            
            # Calculate value (price per point)
            players_df['value'] = players_df['total_points'] / players_df['now_cost']
            
            # Select relevant columns
            players_df = players_df[[
                'web_name', 'element_type', 'team_name', 'now_cost', 
                'total_points', 'minutes', 'goals_scored', 'assists',
                'clean_sheets', 'value', 'selected_by_percent'
            ]]
            
            # Rename columns for clarity
            players_df = players_df.rename(columns={
                'web_name': 'name',
                'element_type': 'position',
                'now_cost': 'price',
                'selected_by_percent': 'ownership'
            })
            
            # Convert position IDs to names
            position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            players_df['position'] = players_df['position'].map(position_map)
            
            # Authenticate and fetch user's actual team
            self.authenticate_with_selenium()
            user_team_data = self.fetch_user_team()

            # Map user team to detailed player info
            current_team = [
                player for player in players_df.to_dict('records')
                if player['id'] in [p['element'] for p in user_team_data['picks']]
            ]
            
            return {
                'players': players_df,
                'teams': teams_df,
                'fixtures': fixtures_df,
                'current_team': current_team,
                'timestamp': pd.Timestamp.now()
            }
        except Exception as e:
            self.logger.error(f"Error scraping data: {e}")
            raise

class TeamOptimizationAgent(FPLAgent):
    def __init__(self):
        super().__init__()
        self.budget = 100.0
        self.position_limits = {
            'GKP': (2, 2),
            'DEF': (5, 5),
            'MID': (5, 5),
            'FWD': (3, 3)
        }

    def _calculate_expected_points(self, players_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.Series:
        try:
            # Simple expected points based on form and fixtures
            points = players_df['total_points'] / players_df['minutes'] * 90
            return points.fillna(0)
        except Exception as e:
            self.logger.error(f"Error calculating expected points: {e}")
            return pd.Series(0, index=players_df.index)

    def _select_optimal_team(self, players_df: pd.DataFrame) -> List[Dict]:
        try:
            selected_team = []
            remaining_budget = self.budget
            
            for position, (min_count, max_count) in self.position_limits.items():
                position_players = players_df[players_df['position'] == position].copy()
                position_players = position_players.nlargest(max_count, 'value')
                
                for _, player in position_players.iterrows():
                    if len(selected_team) < 15 and player['price']/10 <= remaining_budget:
                        selected_team.append(player.to_dict())
                        remaining_budget -= player['price']/10
            
            return selected_team
        except Exception as e:
            self.logger.error(f"Error selecting optimal team: {e}")
            return []

    def _calculate_remaining_budget(self, team: List[Dict]) -> float:
        try:
            spent = sum(player['price']/10 for player in team)
            return self.budget - spent
        except Exception as e:
            self.logger.error(f"Error calculating budget: {e}")
            return 0.0

    async def process(self, data: Dict) -> Dict:
            """
            Optimize and suggest a team based on player and fixture data.
            """
            try:
                self.logger.info("Starting team optimization.")

                # Validate required keys
                if 'players' not in data or 'fixtures' not in data:
                    raise KeyError("Missing required data: 'players' and/or 'fixtures'.")

                players = data['players']
                fixtures = data['fixtures']

                # Example optimization logic: pick top 15 players by total points
                recommended_team = sorted(players, key=lambda x: x.get('total_points', 0), reverse=True)[:15]

                # Construct detailed output
                processed_data = {
                    "recommended_team": recommended_team,
                    "timestamp": datetime.now().isoformat(),
                }

                self.logger.info("Team optimization complete.")
                return processed_data

            except Exception as e:
                self.logger.error(f"Error during team optimization: {e}")
                return {"error": str(e)}
        
class TransferAgent(FPLAgent):
    def __init__(self):
        super().__init__()

    def _analyze_transfers(self, current_team: List[Dict], optimal_team: List[Dict], 
                         available_transfers: int) -> List[Tuple[Dict, Dict]]:
        try:
            if not current_team or not optimal_team:
                return []

            transfers = []
            for curr_player in current_team:
                for opt_player in optimal_team:
                    if (curr_player['position'] == opt_player['position'] and 
                        curr_player['name'] != opt_player['name']):
                        transfers.append((curr_player, opt_player))
                        
            return transfers[:available_transfers]
        except Exception as e:
            self.logger.error(f"Transfer analysis failed: {e}")
            return []

    def _calculate_point_impact(self, transfers: List[Tuple[Dict, Dict]]) -> float:
        try:
            return sum(new['expected_points'] - old['expected_points'] 
                      for old, new in transfers if 'expected_points' in new and 'expected_points' in old)
        except Exception as e:
            self.logger.error(f"Point impact calculation failed: {e}")
            return 0.0

    def _calculate_transfer_cost(self, num_transfers: int, free_transfers: int) -> int:
        try:
            extra_transfers = max(0, num_transfers - free_transfers)
            return extra_transfers * 4
        except Exception as e:
            self.logger.error(f"Transfer cost calculation failed: {e}")
            return 0

    async def process(self, data: Dict) -> Dict:
        try:
            current_team = data.get('current_team', [])
            optimal_team = data.get('optimal_team', [])
            free_transfers = data.get('free_transfers', 1)
            
            transfers = self._analyze_transfers(current_team, optimal_team, free_transfers)
            
            return {
                'suggested_transfers': transfers,
                'expected_point_gain': self._calculate_point_impact(transfers),
                'transfer_cost': self._calculate_transfer_cost(len(transfers), free_transfers)
            }
        except Exception as e:
            self.logger.error(f"Transfer processing failed: {e}")
            return {'suggested_transfers': [], 'expected_point_gain': 0, 'transfer_cost': 0}

class CaptainAgent(FPLAgent):
    def __init__(self):
        super().__init__()

    def _rank_captain_choices(self, team: List[Dict], fixtures: pd.DataFrame) -> List[Dict]:
        try:
            if not team:
                return []
            
            for player in team:
                player['captain_score'] = player.get('expected_points', 0)
            
            return sorted(team, key=lambda x: x.get('captain_score', 0), reverse=True)
        except Exception as e:
            self.logger.error(f"Captain ranking failed: {e}")
            return []

    def _generate_captain_reasoning(self, player: Optional[Dict]) -> str:
        try:
            if not player:
                return "No valid captain choice available"
            return f"Selected based on expected points and fixture difficulty"
        except Exception as e:
            self.logger.error(f"Captain reasoning generation failed: {e}")
            return "Unable to generate reasoning"

    async def process(self, data: Dict) -> Dict:
        try:
            team = data.get('team', [])
            fixtures = data.get('fixtures', pd.DataFrame())
            
            choices = self._rank_captain_choices(team, fixtures)
            
            return {
                'primary_captain': choices[0] if choices else None,
                'vice_captain': choices[1] if len(choices) > 1 else None,
                'alternative_options': choices[2:],
                'reasoning': self._generate_captain_reasoning(choices[0] if choices else None)
            }
        except Exception as e:
            self.logger.error(f"Captain selection failed: {e}")
            return {
                'primary_captain': None,
                'vice_captain': None,
                'alternative_options': [],
                'reasoning': "Captain selection failed"
            }
        
# Orchestrator
class FPLOrchestrator:
    def __init__(self):
        self.data_scraper = DataScraperAgent(email, password)
        self.team_optimizer = TeamOptimizationAgent()
        self.transfer_advisor = TransferAgent()
        self.captain_advisor = CaptainAgent()
        self.logger = logging.getLogger("Orchestrator")

    async def run_workflow(self, user_team_id: int) -> Dict:
        """Run the complete FPL optimization workflow"""
        try:
            # Step 1: Gather Data
            self.logger.info("Starting data collection...")
            raw_data = await self.data_scraper.process({})
            
            # Step 2: Optimize Ideal Team
            self.logger.info("Optimizing ideal team...")
            optimization_result = await self.team_optimizer.process(raw_data)
            
            # Step 3: Generate Transfer Suggestions
            self.logger.info("Analyzing transfers...")
            transfer_suggestions = await self.transfer_advisor.process({
                'current_team': raw_data['current_team'],
                'optimal_team': optimization_result['optimal_team'],
                'free_transfers': raw_data.get('free_transfers', 1)
            })
            
            # Step 4: Captain Selection
            self.logger.info("Selecting captain...")
            captain_advice = await self.captain_advisor.process({
                'team': raw_data['current_team'],
                'fixtures': raw_data['fixtures']
            })
            
            return {
                'optimization_result': optimization_result,
                'transfer_suggestions': transfer_suggestions,
                'captain_advice': captain_advice,
                'analysis_timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            self.logger.error(f"Workflow error: {e}")
            raise

# Usage Example
async def main():
    orchestrator = FPLOrchestrator()
    user_team_id = 12345  # Example team ID
    
    try:
        results = await orchestrator.run_workflow(user_team_id)
        print("Workflow completed successfully!")
        print(f"Suggested transfers: {results['transfer_suggestions']}")
        print(f"Captain recommendation: {results['captain_advice']['primary_captain']}")
    except Exception as e:
        print(f"Error running workflow: {e}")

if __name__ == "__main__":
    asyncio.run(main())