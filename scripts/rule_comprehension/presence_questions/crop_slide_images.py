from PIL import Image
import os

# This script processes the images so they are finalized for the dataset

def crop_image(image_path, im, left, top, right, bottom):
    """
    Crop the image from the left, top, right, and bottom.

    Parameters:
    - image_path: Path to the image file.
    - left: Amount to crop from the left.
    - top: Amount to crop from the top.
    - right: Amount to crop from the right.
    - bottom: Amount to crop from the bottom.
    """

    # TODO, do this in a cleaner way
    aero_list = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 57, 58]
    aero_list = [str(i) + '.jpg' for i in aero_list]
    
    if im in aero_list:
        baseline_im = Image.open('aero.jpg')
    else:
        baseline_im = Image.open('frame.jpg')
    
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
        
        # Append the baseline six view image
        new_width = min(rotated_img.width, baseline_im.width)
        print("Zoomed width ", rotated_img.width)
        print("Zoomed height ", rotated_img.height)
        print("Baseline width ", baseline_im.width)
        print("Baseline height ", baseline_im.height)
        
        baseline_resized_width = rotated_img.width
        baseline_resized_height = int((rotated_img.width/baseline_im.width)*baseline_im.height)
        new_height = baseline_resized_height + rotated_img.height
        
        # new_height = rotated_img.height + baseline_im.height

        # Create a new blank image with the determined size
        new_image = Image.new('RGB', (new_width, new_height))

        # Paste the first image on top
        new_image.paste(rotated_img, (0, 0))
        
        # Resize baseline_im
        baseline_im = baseline_im.resize((baseline_resized_width, baseline_resized_height))

        # Paste the second image below the first one
        new_image.paste(baseline_im, (0, rotated_img.height))

        # Optionally, save the cropped image
        new_image.save('../../../dataset/rule_comprehension/rule_presence_qa/' + image_name + '.jpg')

for im in os.listdir('raw_presence_images'):
    
    # Example usage:
    image_path = 'raw_presence_images/' + im
    crop_image(image_path, im, left=800, top=1800, right=2575, bottom=1775)
