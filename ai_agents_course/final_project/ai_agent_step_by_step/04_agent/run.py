#!/usr/bin/env python3
"""Run Langflow with custom MLE components."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def main():
    """Run Langflow server."""
    import subprocess

    # Set Langflow to use custom components
    os.environ["LANGFLOW_COMPONENTS_PATH"] = str(project_root / "langflow_components")

    # Run Langflow
    cmd = [
        sys.executable,
        "-m",
        "langflow",
        "run",
        "--host", "0.0.0.0",
        "--port", "7860",
    ]

    print("Starting Langflow with MLE components...")
    print(f"Components path: {os.environ['LANGFLOW_COMPONENTS_PATH']}")
    print("UI will be available at: http://localhost:7860")
    print("-" * 50)

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
