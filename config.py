import os
from dotenv import load_dotenv
import certifi

load_dotenv()

# SSL Certificate
os.environ['SSL_CERT_FILE'] = certifi.where()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLAMA_API_KEY = os.getenv('LLAMA_API_KEY')

# OpenAI Model
OPENAI_MODEL = 'gpt-4o-2024-08-06'

#LLaMa Model
LLAMA_MODEL_PATH = 'meta-llama/Llama-3.2-3B'

TRANSFORMERS_CACHE="/n/netscratch/sompolinsky_lab/Lab/lschrage/huggingface_cache"

# Directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


FINAL_DIR = os.path.join(PROJECT_DIR, 'final')
AVG_MANIFOLD_FINAL_DIR = os.path.join(FINAL_DIR, 'average_manifold')

PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
SENTENCES_DIR = os.path.join(DATA_DIR, 'sentences')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
ERROR_DIR = os.path.join(DATA_DIR, 'error')

AVERAGE_MANIFOLD_DIR = os.path.join(PLOTS_DIR, 'average_manifold')
UPDATED_MANIFOLD_DIR = os.path.join(PLOTS_DIR, 'updated_manifold')
COMBINED_DIR = os.path.join(PLOTS_DIR, 'combined')

# Create directories if they don't exist
directories = [FINAL_DIR, AVG_MANIFOLD_FINAL_DIR, DATA_DIR, PLOTS_DIR, SENTENCES_DIR, EMBEDDINGS_DIR, RESULTS_DIR, ERROR_DIR,
              AVERAGE_MANIFOLD_DIR, UPDATED_MANIFOLD_DIR, COMBINED_DIR]

# Create all directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)

MANIFOLD_SENTENCES_FILE = os.path.join(SENTENCES_DIR, 'manifold_sentences.pkl')
TEST_SENTENCES_FILE = os.path.join(SENTENCES_DIR, 'test_sentences.pkl')


CATEGORIES = ["animals", "furniture", "food", "sports", "clothing", 
              "professions", "plants", "electronics", "jewelry", "transportation",
              "music", "beverages", "literature", "countries", "buildings", "work_tools", 
              "body_parts", "games", "weather", "mythical_creatures", "natural_phenomena", 
              "historical_events", "celestial_bodies", "art_movements", "culinary_techniques"]


NUM_SENTENCES_PER_CATEGORY = 200
NUM_TEST_SENTENCES = 100
LAYERS = [0, 5, 10, 15, 20, 25, 32]
M_SHOT = 5
MSHOT_LOOPS = 30