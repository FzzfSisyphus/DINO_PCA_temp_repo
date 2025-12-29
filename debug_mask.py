# debug_masks.py
import pickle
from pathlib import Path

buffer_dir = Path("./replaybuffer")
episode_dirs = sorted([d for d in buffer_dir.iterdir() if d.is_dir()])

for episode_dir in episode_dirs[:3]:
    masks_path = episode_dir / "step0_masks.pkl"
    if masks_path.exists():
        with open(masks_path, "rb") as f:
            masks = pickle.load(f)
        
        print(f"\n=== {episode_dir.name} ===")
        print(f"Type: {type(masks)}")
        
        if isinstance(masks, dict):
            print(f"Keys: {list(masks.keys())}")
            for k, v in masks.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: type={type(v).__name__}, shape={v.shape}")
                elif isinstance(v, list):
                    print(f"  {k}: type=list, len={len(v)}")
                else:
                    print(f"  {k}: type={type(v).__name__}")
        elif isinstance(masks, list):
            print(f"Length: {len(masks)}")
            for i, m in enumerate(masks[:2]):
                if hasattr(m, 'shape'):
                    print(f"  [{i}]: shape={m.shape}")
                elif isinstance(m, dict):
                    print(f"  [{i}]: dict with keys {list(m.keys())}")
        
        break