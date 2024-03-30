# Scaling Robustness

## Installation

- Make sure you have Python version 3.8.5 installed. The code is verified to work with CUDA 11.6 but should work on other versions too.
- Create a virtual environment and install the requirements
```
python -m venv py385
pip install -U pip # upgrade pip to the latest available version
# Download the required torch version
pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

## Directory Structure

- `src/` - Contains the source code files.
- `data/` - Contains the data files.
- `models/` - Contains the trained models.

## Run the Code

1. Navigate to the project directory: `cd /path/to/project`.
2. Activate the virtual environment: `source myenv/bin/activate` (for Linux/Mac) or `myenv\Scripts\activate` (for Windows).
3. Run the code: `python main.py`.
