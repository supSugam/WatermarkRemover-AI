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
        self.inpainting_model = None
        self.inpainting_model_id = None
        self.is_ready = False

    def load_models(self, detection_model_id="florence-community/Florence-2-large", inpainting_model_id="lama"):
        try:
            # Detect if we actually need to load/change detection model
            if self.florence_model is None or detection_model_id not in str(getattr(self.florence_model, 'config', '')):
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

            # Detect if we need to load/change inpainting model
            if self.inpainting_model is None or self.inpainting_model_id != inpainting_model_id:
                print(f"Loading inpainting model: {inpainting_model_id}...", file=sys.stderr)
                
                # Force registration of lama if needed
                if inpainting_model_id == "lama":
                    try:
                        import iopaint.model.lama
                        from iopaint.model_manager import models
                        if "lama" not in models:
                            from iopaint.model.lama import LaMa
                            models["lama"] = LaMa
                    except Exception as reg_err:
                        print(f"Manual lama registration failed: {reg_err}", file=sys.stderr)

                try:
                    self.inpainting_model = ModelManager(name=inpainting_model_id, device=self.device)
                except Exception as e:
                    print(f"Standard ModelManager failed for {inpainting_model_id}: {e}. Retrying with direct class if possible.", file=sys.stderr)
                    from iopaint.model_manager import models
                    if inpainting_model_id in models:
                        model_class = models[inpainting_model_id]
                        self.inpainting_model = model_class(device=self.device)
                    else:
                        raise e
                self.inpainting_model_id = inpainting_model_id

            self.is_ready = True
            return True
        except Exception as e:
            print(f"Error loading models: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
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
            result_np = remwm.process_image_with_lama(np.array(image), np.array(mask), self.inpainting_model)
            inpainted_pil = Image.fromarray(result_np)

            # CRITICAL FIX: The LaMa model subtly alters colors across the whole image
            # when converting back and forth from tensors. We only want to apply the
            # inpainted pixels where the watermark actually was (defined by the mask).
            # So we composite the inpainted image ON TOP OF the original image using the mask.
            final_pil = image.copy()
            final_pil.paste(inpainted_pil, (0, 0), mask)

            # Convert to base64 for fast preview or save to temp?
            # For now, let's return base64
            buffered = BytesIO()
            final_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return {
                "status": "success",
                "image_base64": img_str
            }
        except Exception as e:
            return {"error": str(e)}

def main():
    bridge = WatermarkBridge()
    print(json.dumps({"status": "bridge_started"}))
    sys.stdout.flush()
    
    # Listen for commands
    for line in sys.stdin:
        try:
            line = line.strip()
            if not line: continue
            request = json.loads(line)
            cmd = request.get("command")
            
            if cmd == "load":
                success = bridge.load_models(
                    request.get("detection_model", "florence-community/Florence-2-large"),
                    request.get("inpainting_model", "lama")
                )
                print(json.dumps({"status": "ready" if success else "error", "error": f"Failed to load models" if not success else None}))
            
            elif cmd == "process":
                result = bridge.process_image(
                    request.get("path"),
                    request.get("max_bbox_percent", 10.0),
                    request.get("prompt", "watermark")
                )
                print(json.dumps(result))
            
            elif cmd == "ping":
                print(json.dumps({
                    "status": "pong", 
                    "is_ready": bridge.is_ready,
                    "detection_model": "loaded" if bridge.florence_model else None,
                    "inpainting_model": bridge.inpainting_model_id
                }))
                
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": f"Bridge error: {e}"}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
