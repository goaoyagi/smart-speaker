import sys
import os

# Add src/ to sys.path so that bare imports (e.g. "from listener import Listener")
# resolve correctly when tests import modules via "from src.X import ..."
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
