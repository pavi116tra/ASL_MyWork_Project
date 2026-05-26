# PowerShell script to set up and run the ASL MediaPipe application
# ---------------------------------------------------------------
# This script upgrades pip, installs required Python packages one by one,
# and finally launches the main application.
# It stops execution if any step fails and prints an informative message.
# ---------------------------------------------------------------

# Path to the Python executable (use the default "python" from PATH)
$python = "python"

# Helper function to run a command and check its exit code
function Run-Command($command, $errorMessage) {
    Write-Host "Running: $command"
    Invoke-Expression $command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] $errorMessage (exit code: $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    } else {
        Write-Host "[SUCCESS] $command completed successfully." -ForegroundColor Green
    }
}

# ---------------------------------------------------------------
# 1. Upgrade pip
# ---------------------------------------------------------------
Run-Command "$python -m pip install --upgrade pip" "Failed to upgrade pip."

# ---------------------------------------------------------------
# 2. Install required packages one by one
# ---------------------------------------------------------------
$packages = @(
    "tensorflow==2.12.0",
    "opencv-python",
    "mediapipe",
    "numpy",
    "pillow",
    "h5py"
)

foreach ($pkg in $packages) {
    Run-Command "$python -m pip install $pkg" "Failed to install $pkg."
}

# ---------------------------------------------------------------
# 3. Run the main application
# ---------------------------------------------------------------
$scriptPath = "C:\Users\pavit\Downloads\Gesture_ASL_Regconization-main\Gesture_ASL_Regconization-main\files\asl_app_mediapipe.py"

if (Test-Path $scriptPath) {
    Write-Host "Launching the ASL MediaPipe app..." -ForegroundColor Cyan
    Invoke-Expression "$python `"$scriptPath`""
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] The application exited with errors (exit code: $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    } else {
        Write-Host "[SUCCESS] Application finished without errors." -ForegroundColor Green
    }
} else {
    Write-Host "[ERROR] Cannot find the script at $scriptPath" -ForegroundColor Red
    exit 1
}

# End of script
