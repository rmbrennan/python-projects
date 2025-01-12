import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv
import os

user_team_url = "https://fantasy.premierleague.com/api/bootstrap-static/"

response = requests.get(user_team_url)
response.raise_for_status()
print(response.json())