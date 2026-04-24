$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$pythonCandidates = @(
  ".venv\\Scripts\\python.exe",
  ".venv-1\\Scripts\\python.exe",
  "python"
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
  if ($candidate -ne "python" -and -not (Test-Path $candidate)) {
    continue
  }

  & $candidate -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)" *> $null
  if ($LASTEXITCODE -eq 0) {
    $pythonExe = $candidate
    break
  }
}

if (-not $pythonExe) {
  throw "PyInstaller not found in .venv, .venv-1, or system python. Run: pip install -r requirements.txt"
}

# Build a single-file Windows executable with required model/data dependencies.
& $pythonExe -m PyInstaller --noconfirm --clean --onefile --windowed --name GestureGameController `
  --add-data "model/model.pkl;model" `
  --collect-all mediapipe `
  --collect-all cv2 `
  --collect-all sklearn `
  --collect-all scipy `
  main.py

if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

Write-Host "Build complete. Executable: dist/GestureGameController.exe"
