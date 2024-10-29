from PIL import Image

# Load the original image
source_path = r"C:\Users\Selim\Desktop\Youtube\Video Contents\Turkce\Aktif Dinleme\Thumbnails\Active Listening.jpg"
image = Image.open(source_path)

# Define target thumbnail size and aspect ratio
target_width, target_height = 1280, 720
target_aspect = target_width / target_height

# Get original image size and aspect ratio
original_width, original_height = image.size
original_aspect = original_width / original_height

# Determine the area to crop for a 16:9 aspect ratio
if original_aspect > target_aspect:
    # Crop horizontally if the image is too wide
    new_width = int(target_aspect * original_height)
    left = (original_width - new_width) // 2
    right = left + new_width
    top = 0
    bottom = original_height
else:
    # Crop vertically if the image is too tall
    new_height = int(original_width / target_aspect)
    top = (original_height - new_height) // 2
    bottom = top + new_height
    left = 0
    right = original_width

# Crop and resize the image
cropped_image = image.crop((left, top, right, bottom))
thumbnail_image = cropped_image.resize((target_width, target_height), Image.LANCZOS)

# Save the new thumbnail
save_path = r"C:\Users\Selim\Desktop\Youtube\Video Contents\Turkce\Aktif Dinleme\Thumbnails\Adjusted.jpg"
thumbnail_image.save(save_path)
print(f"Thumbnail saved as '{save_path}'")
