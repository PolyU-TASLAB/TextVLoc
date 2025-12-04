
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import json

# Initialize BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

# Directory containing your images
image_folder = "your_query_folder"  # <- Replace this with your actual folder path
output_file = "captions_output.json"

# List images
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
captions = []

# Process each image and generate caption
for img_file in image_files:
    path = os.path.join(image_folder, img_file)
    image = Image.open(path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(model.device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    captions.append({
        "image": img_file,
        "caption": caption
    })

# Save to JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)

print(f"Saved {len(captions)} captions to {output_file}")
