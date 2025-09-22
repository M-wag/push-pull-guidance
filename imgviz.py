# Module for composing generated images 

import numpy as np
from PIL import Image
import imageio.v2 as imageio
from typing import Literal


#----------------------------------------------------------------------------
# Visualization functions for composing images together.
# With possibility for batching.

def compose_images(images : list[Image.Image], axis=Literal["h", "v"]) -> Image.Image:
    """Takes a list of PIL Images and produces a horizontal composition"""
    if not images:
        raise ValueError("Empty image list provided")

    if axis == "h":
        height = max(image.height for image in images)
        width = sum(image.width for image in images)
        comp = Image.new("RGB", (width, height))

        cum_width = 0
        for image in images:
            comp.paste(image, (cum_width, 0))
            cum_width += image.width

    elif axis == "v":
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        comp = Image.new("RGB", (width, height))

        cum_height = 0
        for image in images:
            comp.paste(image, (0, cum_height))
            cum_height += image.height

    else:
        raise ValueError(f"Invalid axis {axis}, must be 'h' or 'v'")

    return comp
    
def compose_images_batched(image_lists: list[list[Image.Image]], axis=Literal["h", "v"]) -> list[Image.Image]:
    """Batched version of compose_images"""
    if not image_lists:
        raise ValueError("Empty batch list provided")

    # Transpose the 2D list: [[img1a, img1b], [img2a, img2b]] -> [[img1a, img2a], [img1b, img2b]]
    transposed = list(zip(*image_lists))
    return [compose_images(list(group), axis) for group in transposed]

#----------------------------------------------------------------------------
# Create videos showing the progression of edits across different parameter configurations.

def create_edit_videos(original_images: list[Image.Image], 
                      example_images: list[Image.Image], 
                      all_edited_frames: list[list[Image.Image]], 
                      outdir: str, 
                      fps: int = 1) -> None:
    """
    Create videos showing the progression of edits across different parameter configurations.
    
    Args:
        original_images: List of original input images
        example_images: List of example/target images
        all_edited_frames: List of lists where each inner list contains edited images 
                          for a specific parameter configuration
        outdir: Directory to save the output videos
        fps: Frames per second for the output videos
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Create video for each seed
    for seed_idx in range(len(original_images)):
        video_frames = []
        
        # Create a frame for each parameter configuration
        for param_frames in all_edited_frames:
            # Compose the frame: original + edited + example
            frame = compose_images([
                original_images[seed_idx], 
                param_frames[seed_idx], 
                example_images[seed_idx]
            ], "h")
            video_frames.append(np.array(frame))
        
        # Save as video
        video_path = os.path.join(outdir, f"edit_progression_{seed_idx}.mp4")
        imageio.mimwrite(video_path, video_frames, fps=fps, quality=8)
        print(f"Created video for seed {seed_idx} at {video_path}")
