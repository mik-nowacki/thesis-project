import os
import torch

def rescale_bis(input_dir: str):
    """
    Rescales BIS values in all case_*.pt files from [0, 100] to [0, 1]
    by dividing by 100. Files are updated in place.
    """
    files = [f for f in os.listdir(input_dir) if f.startswith('case_') and f.endswith('.pt')]
    
    if not files:
        print(f"No case files found in {input_dir}")
        return
    
    print(f"Found {len(files)} files to process...")

    for fname in files:
        fpath = os.path.join(input_dir, fname)
        
        data = torch.load(fpath, weights_only=False)
        
        # Sanity check before rescaling
        bis = data['bis']
        if bis.max() <= 1.0:
            print(f"[SKIP] {fname} — BIS already in [0,1] range (max={bis.max():.3f})")
            continue
        
        data['bis'] = bis / 100.0
        
        torch.save(data, fpath)
        print(f"[OK] {fname} — BIS rescaled (max was {bis.max():.1f}, now {(bis.max()/100):.3f})")

    print("Done.")


if __name__ == "__main__":
    INPUT_DIR = 'data/processed/patient_dataset'  
    rescale_bis(INPUT_DIR)