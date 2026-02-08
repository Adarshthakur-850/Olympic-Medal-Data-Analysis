import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/cstorm125/information_value/master/data/120-years-of-olympic-history-athletes-and-results/athlete_events.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
