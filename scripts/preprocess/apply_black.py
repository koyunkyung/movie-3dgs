import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def apply_mask_black_bg(image_path: str, mask_path: str, output_path: str):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    # Resize mask if needed
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.NEAREST)
    
    img_array = np.array(image).astype(np.float32)
    mask_array = np.array(mask).astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_array] * 3, axis=-1)
    
    # Black background
    result = img_array * mask_3ch  # Outside mask becomes 0 (black)
    
    Image.fromarray(result.astype(np.uint8)).save(output_path, quality=95)


def process_object(obj_name: str):
    """Apply masks to COLMAP images for one object."""
    colmap_images = Path(f'data/colmap/{obj_name}/images')
    masks_dir = Path(f'data/masks/{obj_name}')
    
    if not colmap_images.exists():
        print(f"  ERROR: {colmap_images} not found")
        return False
    
    if not masks_dir.exists():
        print(f"  ERROR: {masks_dir} not found")
        return False
    
    # Get files
    image_files = sorted(list(colmap_images.glob('*.jpg')) + list(colmap_images.glob('*.png')))
    mask_files = {f.stem: f for f in masks_dir.glob('*.png')}
    
    print(f"\n{obj_name}:")
    print(f"  Images: {len(image_files)}")
    print(f"  Masks: {len(mask_files)}")
    
    count = 0
    for img_file in tqdm(image_files, desc=f"Applying masks to {obj_name}"):
        if img_file.stem in mask_files:
            mask_path = mask_files[img_file.stem]
            # Overwrite the original image
            apply_mask_black_bg(str(img_file), str(mask_path), str(img_file))
            count += 1
    
    print(f"  Applied masks to {count}/{len(image_files)} images")
    return True


def main():
    print("="*60)
    print("Applying SAM masks with BLACK background")
    print("(Image-level preprocessing - allowed by project rules)")
    print("="*60)
    
    # Only process objects, not scene
    objects = ['teapot', 'teddy', 'plant']
    
    for obj_name in objects:
        process_object(obj_name)
    
    print("\n" + "="*60)
    print("NEXT STEP: Re-train 3DGS WITHOUT --white_background flag")
    print("="*60)


if __name__ == '__main__':
    main()