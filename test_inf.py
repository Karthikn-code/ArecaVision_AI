import os
from utils.inference_helper import run_inference

img_dir = r"d:\Areca\archive\Arecanut_dataset\Arecanut_dataset\train\Healthy_Leaf"
files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]
if files:
    print("Testing on:", files[0])
    res = run_inference(files[0], model_name="EfficientNet-B0", use_ensemble=False)
    print("RESULT:")
    print("  Predicted Class:", res["predicted_class"])
    print(f"  Confidence: {res['confidence'] * 100:.2f}%")
    print("  GradCAM path:", res["gradcam_path"])
