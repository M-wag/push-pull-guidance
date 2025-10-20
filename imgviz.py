# Module for composing generated images 

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from typing import Tuple, Optional, Union, Literal


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

#----------------------------------------------------------------------------
# Draw a title top of image


def draw_title(
    im: Image.Image,
    title: str,
    font: Optional[ImageFont.FreeTypeFont] = None,
    padding: int = 8,
    text_fill: Union[Tuple[int,int,int], Tuple[int,int,int,int]] = (255,255,255),
    bg_fill: Union[Tuple[int,int,int,int], Tuple[int,int,int]] = (0,0,0,160),
    position: str = "center"  # "left", "center", "right"
) -> Image.Image:
    """
    Draw `title` on `im` near the top. Returns the image (modified in-place).
    """
    if font is None:
        try:
            # Try a common truetype font (adjust path if needed)
            font = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            font = ImageFont.load_default()

    # Ensure image has alpha channel if we want semi-transparent background
    needs_alpha = isinstance(bg_fill, tuple) and len(bg_fill) == 4 and bg_fill[3] < 255
    if needs_alpha and im.mode != "RGBA":
        im = im.convert("RGBA")

    draw = ImageDraw.Draw(im)

    # Get text bbox (use textbbox if available for more accurate metrics)
    try:
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # fallback
        text_w, text_h = draw.textsize(title, font=font)

    img_w, img_h = im.size

    # Compute x based on requested position
    if position == "left":
        x = padding
    elif position == "right":
        x = img_w - text_w - padding
    else:  # center
        x = (img_w - text_w) // 2

    y = padding

    # Draw background rect behind text (with padding)
    rect_x0 = x - padding
    rect_y0 = y - padding
    rect_x1 = x + text_w + padding
    rect_y1 = y + text_h + padding

    # If bg_fill has alpha but draw.rectangle doesn't accept alpha on non-RGBA images,
    # we ensured image is RGBA above when necessary.
    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=bg_fill)

    # Draw the text
    draw.text((x, y), title, fill=text_fill, font=font)

    return im

#----------------------------------------------------------------------------
# Returns a copy of `im` padded by `top_padding` pixels at the top with `color`.
def pad_top(im: Image.Image, top_padding: int, color=(0, 0, 0)) -> Image.Image:

    width, height = im.size
    new_height = height + top_padding

    # Create a new image with the same width and mode
    new_im = Image.new(im.mode, (width, new_height), color)

    # Paste the original image shifted down
    new_im.paste(im, (0, top_padding))

    return new_im
