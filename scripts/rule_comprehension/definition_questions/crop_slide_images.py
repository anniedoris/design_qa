from PIL import Image
import os

# This script processes the images so they are finalized for the dataset

def crop_image(image_path, left, top, right, bottom):
    """
    Crop the image from the left, top, right, and bottom.

    Parameters:
    - image_path: Path to the image file.
    - left: Amount to crop from the left.
    - top: Amount to crop from the top.
    - right: Amount to crop from the right.
    - bottom: Amount to crop from the bottom.
    """

    # Open the image
    with Image.open(image_path) as img:
        
        # Get the image name
        image_name = image_path.split('/')[1].split('.')[0]
        
        # Calculate the new dimensions
        width, height = img.size
        new_left = left
        new_top = top
        new_right = width - right
        new_bottom = height - bottom

        # Perform the crop
        cropped_img = img.crop((new_left, new_top, new_right, new_bottom))

        # Display the cropped image
        # cropped_img.show()
        
        # Finally, rotate the image by 90 degrees
        rotated_img = cropped_img.rotate(-90, expand=True)

        # Optionally, save the cropped image
        rotated_img.save('../../../dataset/rule_comprehension/rule_definition_qa/' + image_name + '.jpg')

for im in os.listdir('def_slide_images'):
    
    # Example usage:
    image_path = 'def_slide_images/' + im
    crop_image(image_path, left=550, top=600, right=2275, bottom=0)
