#!/usr/bin/env python3
"""
Real-time monitoring dashboard launcher
"""

import subprocess
import sys
import os

def main():
    print("🖥️ Starting Ultra Grid Dashboard...")
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not installed. Installing...")
        subprocess.call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Launch dashboard
    subprocess.call(["streamlit", "run", "dashboard.py"])

if __name__ == "__main__":
    main()