import sys
import argparse
import os
import urllib.request
from huggingface_hub import snapshot_download

def download_model(model_name: str, models_dir: str):
    print(f"Starting download for {model_name}...")
    
    if model_name.lower() == "lama":
        # Torch hub looks in $TORCH_HOME/hub/checkpoints
        # Since we set TORCH_HOME to the models_dir, we put it in models_dir/hub/checkpoints
        lama_dir = os.path.join(models_dir, "hub", "checkpoints")
        os.makedirs(lama_dir, exist_ok=True)
        lama_file = os.path.join(lama_dir, "big-lama.pt")
        
        print("Downloading LaMA model (~196MB)...")
        # Ensure we stream progress to stdout for Rust bridge to read
        def report_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                if percent % 10 == 0:
                    sys.stdout.write(f"Downloading LaMA: {min(percent, 100)}%\n")
                    sys.stdout.flush()
                    
        urllib.request.urlretrieve(
            "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt", 
            lama_file,
            reporthook=report_hook
        )
        print("\nDownload complete!")
    else:
        # Assumed florence model
        m_id = f"florence-community/{model_name.replace('florence-2-', 'Florence-2-')}"
        print(f"Downloading {m_id} via HuggingFace Hub...")
        snapshot_download(m_id, local_dir_use_symlinks=False)
        print("Download complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models for WatermarkRemover-AI")
    parser.add_argument("--model", type=str, required=True, help="Model ID (lama, florence-2-base, florence-2-large)")
    args = parser.parse_args()
    
    # Honor environmental variables passed from Rust (like HF_HOME, TORCH_HOME)
    models_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    download_model(args.model, models_dir)
