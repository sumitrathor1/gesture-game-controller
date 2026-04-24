# Gesture Game Controller

Control simple games with hand gestures using a webcam, MediaPipe, and a trained scikit-learn model.

## Requirements

- Windows 10/11
- Python 3.10
- Webcam

## Setup (Source Run)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If your model is not available yet, train first:

```powershell
python train_model.py
```

Then run:

```powershell
python main.py
```

## Build EXE (Windows)

Install dependencies (includes `pyinstaller`):

```powershell
pip install -r requirements.txt
```

Build:

```powershell
./build_exe.ps1
```

Output executable:

- `dist/GestureGameController.exe`

## Smoke Test EXE

Run the EXE directly:

```powershell
./dist/GestureGameController.exe
```

If Windows SmartScreen appears, click **More info** -> **Run anyway**.

## Notes

- The app needs `model/model.pkl` at build time; the build script bundles it into the EXE.
- Press `Esc` in the camera window to close it.
