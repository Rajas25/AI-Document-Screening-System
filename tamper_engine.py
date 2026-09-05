import os
import cv2
import numpy as np
from PIL import Image, ImageChops

def analyze_metadata(image_bytes) -> tuple:
    """Scans binary headers for digital editing software signatures (Photoshop/Canva)."""
    try:
        content = str(image_bytes).lower()
        editing_tools = ["adobe", "photoshop", "gimp", "canva", "picsart"]
        detected_tools = [tool for tool in editing_tools if tool in content]
        
        if detected_tools:
            return True, f"Metadata Alteration Trace: Edited via {', '.join(detected_tools)}"
        return False, "Clean metadata structure."
    except Exception:
        return False, "Metadata scan omitted."

def check_text_manipulation(image_path, quality=90) -> float:
    """
    Performs Error Level Analysis (ELA). 
    Resaving at a specific quality level highlights anomalous high-frequency pixel changes 
    indicative of digitally replaced text blocks.
    """
    try:
        temp_file = "temp_ela.jpg"
        original = Image.open(image_path).convert('RGB')
        
        # Save at target compression quality
        original.save(temp_file, 'JPEG', quality=quality)
        temporary = Image.open(temp_file)
        
        # Calculate absolute pixel variance differences
        ela_image = ImageChops.difference(original, temporary)
        
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        # Clean up disk file footprint
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return float(max_diff) # High variance value indicates potential text/photo replacement
    except Exception:
        return 0.0
