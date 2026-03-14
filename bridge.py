import sys
import json
import base64
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModel
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config

# Add current dir to path to import remwm
sys.path.append(str(Path(__file__).parent))
import remwm

class WatermarkBridge:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.florence_model = None
        self.florence_processor = None
        self.lama_model = None
        self.is_ready = False

    def load_models(self, detection_model_id="florence-community/Florence-2-large"):
        try:
            print(f"Loading detection model: {detection_model_id}...", file=sys.stderr)
            # Try native ImageTextToText first (Transformers 4.42+), fallback to CausalLM (Remote Code / Old Versions), then base AutoModel
            try:
                self.florence_model = AutoModelForImageTextToText.from_pretrained(detection_model_id, trust_remote_code=True).to(self.device).eval()
            except Exception:
                try:
                    self.florence_model = AutoModelForCausalLM.from_pretrained(detection_model_id, trust_remote_code=True).to(self.device).eval()
                except Exception:
                    self.florence_model = AutoModel.from_pretrained(detection_model_id, trust_remote_code=True).to(self.device).eval()
            
            self.florence_processor = AutoProcessor.from_pretrained(detection_model_id, trust_remote_code=True)
            
            print("Loading inpainting model: lama...", file=sys.stderr)
            self.lama_model = ModelManager(name="lama", device=self.device)
            self.is_ready = True
            return True
        except Exception as e:
            print(f"Error loading models: {e}", file=sys.stderr)
            return False

    def process_image(self, image_path, max_bbox_percent=10.0, detection_prompt="watermark"):
        if not self.is_ready:
            return {"error": "Models not loaded"}

        try:
            image = Image.open(image_path).convert("RGB")
            mask = remwm.get_watermark_mask(
                image, 
                self.florence_model, 
                self.florence_processor, 
                self.device, 
                max_bbox_percent, 
                detection_prompt
            )

            # Check if watermark was detected
            if mask.getextrema()[1] == 0:
                return {"status": "skipped", "message": "No watermark detected"}

            # Inpaint
            result_np = remwm.process_image_with_lama(np.array(image), np.array(mask), self.lama_model)
            result_pil = Image.fromarray(result_np)

            # Convert to base64 for fast preview or save to temp?
            # For now, let's return base64
            buffered = BytesIO()
            result_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return {
                "status": "success",
                "image_base64": img_str
            }
        except Exception as e:
            return {"error": str(e)}

def main():
    bridge = WatermarkBridge()
    
    # Listen for commands
    for line in sys.stdin:
        try:
            request = json.loads(line)
            cmd = request.get("command")
            
            if cmd == "load":
                success = bridge.load_models(request.get("model_id", "florence-community/Florence-2-large"))
                print(json.dumps({"status": "ready" if success else "error"}))
            
            elif cmd == "process":
                result = bridge.process_image(
                    request.get("path"),
                    request.get("max_bbox_percent", 10.0),
                    request.get("prompt", "watermark")
                )
                print(json.dumps(result))
            
            elif cmd == "ping":
                print(json.dumps({"status": "pong"}))
                
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": f"Bridge error: {e}"}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
