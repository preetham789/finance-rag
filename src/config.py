# src/config.py
"""
Central config -- every module imports from here.
No magic strings scattered across files.

All tuneable parameters are readable from environment variables so you can
override them without editing code (e.g. in .env or CI).
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()  # reads your .env file automatically

# -- Project root (works regardless of where you run the script from) --
ROOT_DIR = Path(__file__).parent.parent

# -- Data paths --
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
TEST_SETS_DIR  = DATA_DIR / "test_sets"

# -- Experiment paths --
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
CHROMA_DIR      = EXPERIMENTS_DIR / "chromadb"
MLFLOW_DIR      = EXPERIMENTS_DIR / "mlflow"

# -- API Keys (fail loudly if missing -- better than silent errors) --
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "").strip()

# -- Chunking defaults (you'll experiment with these in Phase 5) --
CHUNK_SIZE    = 1024
CHUNK_OVERLAP = 128

# -- Retrieval defaults --
TOP_K = 5       # number of chunks to retrieve per query

# -- Retrieval quality gate --
# Pre-LLM refusal threshold: if the best chunk score is below this value,
# return a refusal without calling the LLM (saves tokens, avoids hallucination).
# Override at runtime via the RETRIEVAL_THRESHOLD env var.
RETRIEVAL_THRESHOLD: float = float(os.getenv("RETRIEVAL_THRESHOLD", "0.20"))

# -- Hybrid blend weight --
# Relative weight of dense vs BM25 scores (1.0 = pure dense, 0.0 = pure BM25).
# Override at runtime via the HYBRID_ALPHA env var.
HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.7"))
