# Getting Started

This guide will walk you through cloning the repository and setting up your environment to run the tutorials.

!!! warning "Prerequisites"
    Before starting, ensure you've completed the [Installation Guide](installation.md) to install:
    
    - astral-uv
    - Jupyter Notebook/Lab
    - ipykernel

---

## Step 1: Clone the Repository

First, clone the tutorial repository to your local machine.

=== "HTTPS"

    ```bash
    git clone https://github.com/yourusername/generative-tutorials.git
    cd generative-tutorials
    ```

=== "SSH"

    ```bash
    git clone git@github.com:yourusername/generative-tutorials.git
    cd generative-tutorials
    ```

=== "GitHub CLI"

    ```bash
    gh repo clone yourusername/generative-tutorials
    cd generative-tutorials
    ```

---

## Step 2: Choose Your Setup Method

You can run the tutorials using either:

1. **astral-uv** (Recommended) - Modern, fast package manager
2. **Traditional pip** - Classic Python package management

Choose the method that works best for you.

---

## Method 1: Setup with astral-uv (Recommended)

astral-uv is faster and more reliable than traditional pip.

### Navigate to a Tutorial

Each tutorial is in its own directory:

```bash
cd tutorial_1_ddpm        # For Tutorial 1
# OR
cd tutorial_2_flow_matching   # For Tutorial 2
```

### Create Virtual Environment

```bash
uv venv .venv
```

### Activate the Environment

=== "Linux / macOS"

    ```bash
    source .venv/bin/activate
    ```

=== "Windows (PowerShell)"

    ```powershell
    .venv\Scripts\Activate.ps1
    ```

=== "Windows (Command Prompt)"

    ```cmd
    .venv\Scripts\activate.bat
    ```

### Install Dependencies

Each tutorial has a `pyproject.toml` file that defines its dependencies.

```bash
# Install from pyproject.toml
uv pip install -e .

# OR install directly
uv pip install torch torchvision torchaudio numpy matplotlib tqdm
```

### Register Jupyter Kernel

```bash
uv pip install ipykernel
python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
```

### Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
jupyter kernelspec list
```

---

## Method 2: Setup with Traditional pip

If you prefer traditional pip, you can use it instead.

### Navigate to a Tutorial

```bash
cd tutorial_1_ddpm        # For Tutorial 1
# OR
cd tutorial_2_flow_matching   # For Tutorial 2
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment

=== "Linux / macOS"

    ```bash
    source .venv/bin/activate
    ```

=== "Windows (PowerShell)"

    ```powershell
    .venv\Scripts\Activate.ps1
    ```

=== "Windows (Command Prompt)"

    ```cmd
    .venv\Scripts\activate.bat
    ```

### Install Dependencies

```bash
# Install from pyproject.toml
pip install -e .

# OR install directly
pip install torch torchvision torchaudio numpy matplotlib tqdm
```

### Register Jupyter Kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
```

### Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
jupyter kernelspec list
```

---

## Step 3: Running the Tutorials

You have two options for running the tutorials:

### Option A: Run in Terminal (CLI)

Each tutorial can be run as a Python script:

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\Activate.ps1  # Windows

# Run the main script
python -m flow_matching_tutorial.main

# OR run directly
python flow_matching_tutorial/main.py
```

**Outputs will be saved to the `outputs/` directory.**

### Option B: Run in Jupyter Notebook/Lab

**Start Jupyter:**

```bash
# Start Jupyter Notebook
jupyter notebook

# OR start JupyterLab
jupyter lab
```

**Open the tutorial notebook:**

1. Navigate to the tutorial directory in Jupyter
2. Open `tutorial_notebook_Flow.ipynb` (or the appropriate notebook)
3. Select the "Tutorial Environment" kernel from the kernel menu
4. Run the cells!

### Option C: Run in VSCode

1. Open VSCode in the repository directory:
   ```bash
   code .
   ```

2. Open the notebook file (e.g., `tutorial_2_flow_matching/tutorial_notebook_Flow.ipynb`)

3. Click on the kernel selector in the top-right corner

4. Select "Tutorial Environment"

5. Run the cells using the run button or `Shift+Enter`

!!! tip "Recommended Workflow"
    - **Learning**: Use Jupyter Notebook/Lab for interactive exploration
    - **Experimentation**: Use VSCode for code editing and debugging
    - **Automation**: Use CLI scripts for batch processing

---

## Repository Structure

Here's an overview of the repository structure:

```
generative-tutorials/
├── docs/                      # Documentation (this website)
├── tutorial_1_ddpm/          # Tutorial 1: DDPM
│   ├── ddpm_tutorial/        # Python package
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── diffusion.py
│   │   ├── utils.py
│   │   └── visualization.py
│   ├── outputs/              # Generated outputs
│   ├── pyproject.toml        # Dependencies
│   └── tutorial_notebook_DDPM.ipynb
│
├── tutorial_2_flow_matching/ # Tutorial 2: Flow Matching
│   ├── flow_matching_tutorial/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── flow.py
│   │   ├── flow_solutions.py
│   │   ├── utils.py
│   │   └── visualization.py
│   ├── outputs/              # Generated outputs
│   ├── pyproject.toml
│   └── tutorial_notebook_Flow.ipynb
│
├── tutorial_3_advanced/      # Tutorial 3 (Coming Soon)
├── tutorial_4_score_based/   # Tutorial 4 (Coming Soon)
├── tutorial_5_applications/  # Tutorial 5 (Coming Soon)
├── mkdocs.yml                # Documentation config
├── README.md
└── LICENSE
```

---

## Running Multiple Tutorials

Each tutorial has its own virtual environment and kernel. To run multiple tutorials:

**Setup Tutorial 1:**

```bash
cd tutorial_1_ddpm
uv venv .venv
source .venv/bin/activate
uv pip install -e .
python -m ipykernel install --user --name=tutorial1 --display-name="Tutorial 1 - DDPM"
```

**Setup Tutorial 2:**

```bash
cd ../tutorial_2_flow_matching
uv venv .venv
source .venv/bin/activate
uv pip install -e .
python -m ipykernel install --user --name=tutorial2 --display-name="Tutorial 2 - Flow Matching"
```

**Now you can switch between kernels in Jupyter!**

---

## Common Workflows

### Daily Workflow

```bash
# Activate environment
cd tutorial_2_flow_matching
source .venv/bin/activate

# Pull latest changes
git pull

# Start Jupyter
jupyter lab

# Work on notebooks...

# Deactivate when done
deactivate
```

### Running Experiments

```bash
# Activate environment
source .venv/bin/activate

# Modify config in main.py
nano flow_matching_tutorial/main.py

# Run experiment
python -m flow_matching_tutorial.main

# Check outputs
ls outputs/
```

### Updating Dependencies

```bash
# Activate environment
source .venv/bin/activate

# Update with uv
uv pip install --upgrade torch numpy matplotlib

# OR update with pip
pip install --upgrade torch numpy matplotlib
```

---

## GPU Support (Optional)

If you have an NVIDIA GPU and want to use it:

### Check CUDA Availability

```bash
nvidia-smi
```

### Install PyTorch with CUDA Support

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to get the right command for your system.

Example for CUDA 12.1:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

!!! note "CPU is Fine!"
    All tutorials work on CPU. GPU just makes them faster. The 2D toy datasets run quickly on CPU.

---

## Next Steps

Now that you're set up, choose a tutorial to start:

- [Tutorial 1: DDPM](tutorials/tutorial-1.md) - Start with diffusion models
- [Tutorial 2: Flow Matching](tutorials/tutorial-2.md) - Learn ODE-based generation

---

## Quick Reference

**Activate environment:**

```bash
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\Activate.ps1  # Windows
```

**Run notebook:**

```bash
jupyter lab
```

**Run CLI:**

```bash
python -m flow_matching_tutorial.main
```

**Update kernel:**

```bash
python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
```

---

## Troubleshooting

### Module not found

**Problem:** `ModuleNotFoundError: No module named 'flow_matching_tutorial'`

**Solution:** Install the package in editable mode:

```bash
source .venv/bin/activate
uv pip install -e .
```

### Kernel not showing in Jupyter

**Problem:** Can't see "Tutorial Environment" in kernel list

**Solution:** Reinstall the kernel:

```bash
python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
```

### Import errors in notebook

**Problem:** `ImportError: cannot import name 'something'`

**Solution:** Make sure you selected the correct kernel in Jupyter

### CUDA out of memory

**Problem:** GPU memory error

**Solution:** Use CPU instead or reduce batch size in config

For more help, see [Troubleshooting](troubleshooting.md).
