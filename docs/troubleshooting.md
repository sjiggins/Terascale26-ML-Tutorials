# Troubleshooting

Common issues and their solutions.

---

## Installation Issues

### uv not found after installation

**Error:**
```bash
bash: uv: command not found
```

**Solution:**

=== "Linux / macOS"
    Add uv to your PATH:
    ```bash
    export PATH="$HOME/.cargo/bin:$PATH"
    source ~/.bashrc  # or ~/.zshrc
    ```

=== "Windows"
    Add `C:\Users\<YourUsername>\.cargo\bin` to your PATH environment variable, then restart your terminal.

---

### Permission denied during installation

**Error:**
```bash
Permission denied: /usr/local/bin/uv
```

**Solution:**

Don't use `sudo`. Install in user space:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with pip:

```bash
pip install --user uv
```

---

### Python version too old

**Error:**
```bash
Python 3.8 is not supported. Please use Python 3.9+
```

**Solution:**

Install a newer Python version:

=== "Linux"
    ```bash
    # Using deadsnakes PPA (Ubuntu/Debian)
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.11
    ```

=== "macOS"
    ```bash
    # Using Homebrew
    brew install python@3.11
    ```

=== "Windows"
    Download from [python.org](https://www.python.org/downloads/)

---

## Environment Setup Issues

### Virtual environment not activating

**Error:**
```bash
.venv/bin/activate: No such file or directory
```

**Solution:**

Create the virtual environment first:

```bash
cd tutorial_2_flow_matching
uv venv .venv
source .venv/bin/activate  # Linux/macOS
```

On Windows:
```powershell
.venv\Scripts\Activate.ps1
```

---

### Module not found after installation

**Error:**
```python
ModuleNotFoundError: No module named 'flow_matching_tutorial'
```

**Solution:**

Install the package in editable mode:

```bash
source .venv/bin/activate
uv pip install -e .
```

Or install directly:

```bash
uv pip install torch numpy matplotlib
```

---

## Jupyter Issues

### Kernel not showing in Jupyter

**Error:**

Can't find "Tutorial Environment" in kernel list.

**Solution:**

Register the kernel:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name=tutorial-env --display-name="Tutorial Environment"
```

Verify it's registered:

```bash
jupyter kernelspec list
```

---

### Wrong kernel selected

**Error:**

Imports work in terminal but not in notebook.

**Solution:**

1. Click "Kernel" → "Change Kernel" in Jupyter
2. Select "Tutorial Environment"
3. Restart the kernel

Or in VSCode:
1. Click kernel selector (top-right)
2. Select "Tutorial Environment"

---

### Jupyter won't start

**Error:**
```bash
jupyter: command not found
```

**Solution:**

Install Jupyter:

```bash
source .venv/bin/activate
uv pip install jupyter notebook jupyterlab
```

Or use a dedicated Jupyter environment:

```bash
uv venv ~/.jupyter-env
source ~/.jupyter-env/bin/activate
uv pip install jupyter notebook jupyterlab
```

---

## PyTorch Issues

### CUDA not available

**Error:**
```python
RuntimeError: CUDA out of memory
```

**Or:**
```python
torch.cuda.is_available()  # Returns False
```

**Solution:**

1. **Use CPU instead:**
   ```python
   device = "cpu"
   ```

2. **Install CUDA-enabled PyTorch:**
   
   Visit [pytorch.org](https://pytorch.org/get-started/locally/) and get the right command.
   
   Example for CUDA 12.1:
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```

---

### Dtype mismatch errors

**Error:**
```python
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
```

**Solution:**

This is fixed in the latest version. Update your tutorial files, or explicitly cast to float32:

```python
x = torch.tensor(data, dtype=torch.float32)
```

---

## Runtime Errors

### Out of memory

**Error:**
```python
RuntimeError: CUDA out of memory
```

**Or:**
```python
MemoryError: Unable to allocate array
```

**Solution:**

1. **Reduce batch size:**
   ```python
   config["batch_size"] = 64  # Instead of 256
   ```

2. **Use CPU:**
   ```python
   device = "cpu"
   ```

3. **Close other programs**

4. **Simplify model:**
   ```python
   config["hidden_dim"] = 64  # Instead of 128
   ```

---

### Training loss not decreasing

**Problem:**

Loss stays high or increases during training.

**Solution:**

1. **Check learning rate:**
   ```python
   config["learning_rate"] = 1e-4  # Try lower
   ```

2. **Verify data normalization:**
   ```python
   data = (data - data.mean()) / data.std()
   ```

3. **Check TODO implementations:**
   - Review your code in `flow.py`
   - Compare with `flow_solutions.py`

4. **Train longer:**
   ```python
   config["n_epochs"] = 200
   ```

5. **Verify loss computation:**
   - Add print statements
   - Check gradients aren't NaN

---

### Samples look like noise

**Problem:**

Generated samples don't resemble training data.

**Solution:**

1. **Use more sampling steps:**
   ```python
   config["n_euler_steps"] = 200  # Instead of 100
   ```

2. **Train longer:**
   - Check if loss converged
   - Plot training curve

3. **Verify sampling procedure:**
   - Check ODE solver implementation
   - Try different solver (Euler vs RK45)

4. **Check model predictions:**
   ```python
   # Visualize intermediate steps
   visualize_reverse_process_steps(trajectory)
   ```

---

## Import Errors

### Circular import

**Error:**
```python
ImportError: cannot import name 'X' from partially initialized module
```

**Solution:**

1. **Check for circular dependencies** in your imports

2. **Use local imports:**
   ```python
   def function():
       from flow_matching_tutorial import something
       # Use it here
   ```

3. **Reorganize code** to avoid circles

---

### Attribute not found

**Error:**
```python
AttributeError: 'numpy.ndarray' object has no attribute 'cpu'
```

**Solution:**

This is fixed in the latest version. Update `visualization.py`, or add type checking:

```python
if torch.is_tensor(data):
    data_np = data.cpu().numpy()
else:
    data_np = data
```

---

## Visualization Issues

### Plots not showing

**Problem:**

`plt.show()` doesn't display plots.

**Solution:**

In Jupyter, use:

```python
%matplotlib inline
```

Or for interactive plots:

```python
%matplotlib widget
```

---

### Animation not saving

**Error:**
```python
FileNotFoundError: [Errno 2] No such file or directory: 'outputs/animation.gif'
```

**Solution:**

Create the outputs directory:

```python
import os
os.makedirs("outputs", exist_ok=True)
```

---

### Images look weird

**Problem:**

Generated images are blank, all black, or distorted.

**Solution:**

1. **Check data range:**
   ```python
   print(f"Data range: {data.min()} to {data.max()}")
   ```

2. **Normalize properly:**
   ```python
   data = (data - data.min()) / (data.max() - data.min())
   ```

3. **Verify model output range:**
   ```python
   output = torch.clamp(output, 0, 1)
   ```

---

## Platform-Specific Issues

### Windows: PowerShell execution policy

**Error:**
```powershell
.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled
```

**Solution:**

Run PowerShell as Administrator and execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then restart PowerShell and try again.

---

### macOS: SSL certificate error

**Error:**
```bash
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solution:**

Install certificates:

```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```

Or:

```bash
pip install --upgrade certifi
```

---

### Linux: libGL error

**Error:**
```bash
libGL error: failed to load driver
```

**Solution:**

Install mesa libraries:

```bash
sudo apt install libgl1-mesa-glx  # Ubuntu/Debian
```

---

## Performance Issues

### Slow training on CPU

**Problem:**

Training takes too long on CPU.

**Solution:**

1. **Use GPU** if available

2. **Reduce data size:**
   ```python
   config["n_samples"] = 5000  # Instead of 10000
   ```

3. **Use smaller model:**
   ```python
   config["hidden_dim"] = 64
   config["n_layers"] = 2
   ```

4. **Fewer epochs:**
   ```python
   config["n_epochs"] = 50
   ```

---

### Slow sampling

**Problem:**

Generating samples takes too long.

**Solution:**

1. **Use RK45 instead of Euler:**
   ```python
   samples = flow.sample(model, method="rk45")
   ```

2. **Reduce sampling steps:**
   ```python
   config["n_euler_steps"] = 50  # Instead of 100
   ```

3. **Use GPU**

4. **Generate fewer samples:**
   ```python
   config["n_samples_to_generate"] = 100
   ```

---

## Still Having Issues?

If your problem isn't listed here:

1. **Check the [FAQ](faq.md)**

2. **Search GitHub issues:**
   [github.com/yourusername/generative-tutorials/issues](https://github.com/yourusername/generative-tutorials/issues)

3. **Open a new issue** with:
   - Error message (full traceback)
   - Steps to reproduce
   - System information:
     ```bash
     python --version
     pip list | grep torch
     uname -a  # Linux/macOS
     ```

4. **Ask in Discussions:**
   [github.com/yourusername/generative-tutorials/discussions](https://github.com/yourusername/generative-tutorials/discussions)

---

## Debugging Tips

### Enable verbose logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check tensor shapes

```python
print(f"x shape: {x.shape}")
print(f"t shape: {t.shape}")
```

### Validate gradients

```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

### Profile code

```python
import time
start = time.time()
# ... code ...
print(f"Time: {time.time() - start:.2f}s")
```

---

## Quick Fixes Checklist

Before opening an issue, try:

- [ ] Restart kernel/Python
- [ ] Deactivate and reactivate virtual environment
- [ ] Reinstall packages: `uv pip install -e .`
- [ ] Clear Jupyter outputs and restart kernel
- [ ] Update to latest code: `git pull`
- [ ] Check you're in the right directory
- [ ] Verify Python version: `python --version`
- [ ] Check imports work in Python REPL
- [ ] Read error message carefully

---

Happy troubleshooting! Most issues have simple fixes.
