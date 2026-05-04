# scripts/setup_dirs.py

"""
Run once after cloning. Creates all data/experiment directories.
Why: data/ is gitignored, so collaborators need to recreate it.
"""

import sys
from pathlib import Path
   

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add project root to path
from src.config import RAW_DIR, PROCESSED_DIR, TEST_SETS_DIR, CHROMA_DIR, MLFLOW_DIR

dirs = [RAW_DIR, PROCESSED_DIR, TEST_SETS_DIR, CHROMA_DIR, MLFLOW_DIR]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {d}")  
      
print("All directories set up! You're good to go.")
