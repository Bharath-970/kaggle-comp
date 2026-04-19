import base64
import json
import os
import zipfile
from io import BytesIO

def bundle():
    print("📦 Bundling repository for Kaggle...")
    
    # 1. Create In-Memory Zip
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Bundle src/
        for root, dirs, files in os.walk('src'):
            if '__pycache__' in dirs: dirs.remove('__pycache__')
            for file in files:
                if file.endswith('.py'):
                    path = os.path.join(root, file)
                    zf.write(path)
        
        # Bundle specific scripts
        scripts_to_include = ['kaggle_solve.py', 'ideas_stack.py', 'search_solvers.py']
        for s in scripts_to_include:
            path = os.path.join('scripts', s)
            if os.path.exists(path):
                zf.write(path)
    
    buf.seek(0)
    b64_data = base64.b64encode(buf.read()).decode('utf-8')
    print(f"✅ Bundle created. Size (encoded): {len(b64_data) / 1024:.1f} KB")

    # 2. Build Notebook JSON
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# NeuroGolf V12: Zero-Dependency Master Sweep\n",
                    "\n",
                    "This notebook contains the entirely self-contained NeuroGolf engine. \n",
                    "It bypasses all dataset mount issues by embedding the source code as an encoded bundle."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Install missing dependencies for ONNX export\n",
                    "!pip install onnxscript onnx --quiet\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import base64\n",
                    "import zipfile\n",
                    "import os\n",
                    "from pathlib import Path\n",
                    "\n",
                    f"bundle_b64 = \"{b64_data}\"\n",
                    "\n",
                    "print(\"🛠️ Extracting NeuroGolf Master Bundle...\")\n",
                    "try:\n",
                    "    with open(\"bundle.zip\", \"wb\") as f:\n",
                    "        f.write(base64.b64decode(bundle_b64))\n",
                    "    \n",
                    "    with zipfile.ZipFile(\"bundle.zip\", \"r\") as zip_ref:\n",
                    "        zip_ref.extractall(\".\")\n",
                    "    \n",
                    "    # Validation (Brutal Truth Safeguard)\n",
                    "    expected_files = [\"src/neurogolf/constants.py\", \"scripts/kaggle_solve.py\"]\n",
                    "    for f in expected_files:\n",
                    "        if not os.path.exists(f):\n",
                    "            raise RuntimeError(f\"❌ Critical file missing after extraction: {f}\")\n",
                    "    \n",
                    "    print(\"✅ Extraction Successful. Library validated.\")\n",
                    "    os.remove(\"bundle.zip\")\n",
                    "except Exception as e:\n",
                    "    raise RuntimeError(f\"💥 Bundle extraction failed: {str(e)}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "sys.path.insert(0, os.path.abspath('src'))\n",
                    "sys.path.insert(0, os.path.abspath('scripts'))\n",
                    "\n",
                    "from scripts import kaggle_solve\n",
                    "\n",
                    "print(\"🚀 Starting 400-Task Sweep...\")\n",
                    "kaggle_solve.main()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    output_path = 'scripts/kaggle_deployment/neurogolf_v12_bundle.ipynb'
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"🚀 Master Notebook generated at: {output_path}")

if __name__ == "__main__":
    bundle()
