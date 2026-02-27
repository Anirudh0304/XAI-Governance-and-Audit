import sys
from pathlib import Path
import subprocess

# Add project root to Python path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# Run Streamlit
subprocess.run(["streamlit", "run", "src/dashboard.py"])