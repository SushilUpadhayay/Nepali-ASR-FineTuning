from pathlib import Path
root = Path("D:/NemoASR/pilot_run")
files = ["pilot_run_manifest.json", "pilot_run_train_manifest.json", "pilot_run_val_manifest.json"]
old = "D:/NemoASR/pilot_run/audio_files"
new = "D:/NemoASR/pilot_run/audio_files/"
for name in files:
    path = root / name
    if path.exists():
        text = path.read_text()
        updated = text.replace(old, new)
        if updated != text:
            path.write_text(updated)
            print(f"Updated {path}")
        else:
            print(f"No changes needed for {path}")
    else:
        print(f"Missing {path}")