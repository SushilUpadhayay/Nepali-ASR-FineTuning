from pathlib import Path
root = Path("/home/nextai/asr-nepali/Nepali-ASR-FineTuning/tuning_run")
files = ["manifest.json", "train_manifest.json", "val_manifest.json"]
old = "d:/FineTuning/tuning_run/tuning_audio/"
new = "/home/nextai/asr-nepali/Nepali-ASR-FineTuning/tuning_run/tuning_audio/"
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