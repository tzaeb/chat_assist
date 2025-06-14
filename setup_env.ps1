# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Note: Activation in a script is tricky,
# subsequent commands will be executed in the context of the script's session,
# not the activated venv directly if you just call .venv\Scripts\Activate.ps1)
# So, we will call python and pip from within the venv directly.

# Update pip and setuptools
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools

# Install requirements
.\.venv\Scripts\pip.exe install -r requirements.txt

Write-Host "Virtual environment '.venv' created and dependencies installed."
Write-Host "To activate the virtual environment in your current session, run: .\.venv\Scripts\Activate.ps1"
