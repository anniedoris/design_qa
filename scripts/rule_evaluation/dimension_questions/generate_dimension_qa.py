from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import pandas as pd
sys.path.append("..")
from common_prompts import prompt_preamble

def convert_pdf_to_images(pdf_path):
    """
    Convert single page PDF to an image.

    Parameters:
    - pdf_path: Path to the PDF file.
    """
    # Convert PDF to a list of images
    images = convert_from_path(pdf_path, dpi=600)
    
    for i, image in enumerate(images):
        pass
    # image.save('dimension_jpg/' + pdf_path)
    
    return image

def crop_image(img, left, top, right, bottom):
    # Rotate and crop the baseline image
    # Calculate the new dimensions
    width, height = img.size
    new_left = left
    new_top = top
    new_right = width - right
    new_bottom = height - bottom

    # Perform the crop
    cropped_img = img.crop((new_left, new_top, new_right, new_bottom))
    
    return cropped_img

def rotate_image(img, rotation_angle):
    rotated_img = img.rotate(rotation_angle, expand=True)
    return rotated_img

def concatenate_images(base_img, zoomed_img):
    print(base_img.width, base_img.height)
    print(zoomed_img.width, zoomed_img.height)
    concatenated_width = zoomed_img.width
    resized_base_height = int((base_img.height / base_img.width) * zoomed_img.width)
    concat_image = Image.new('RGB', (concatenated_width, zoomed_img.height + resized_base_height))
    
    # Paste the first image on top
    concat_image.paste(zoomed_img, (0, 0))
    
    # Resize baseline_im
    baseline_resized = base_img.resize((concatenated_width, resized_base_height))

    # Paste the second image below the first one
    concat_image.paste(baseline_resized, (0, zoomed_img.height))
    
    return concat_image

def draw_line_img(img, y, thickness, offset):
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Calculate line coordinates
    line_start = (offset, y)
    line_end = (img.width - offset, y)
    
    # Draw the horizontal line
    draw.line([line_start, line_end], fill='black', width=thickness)
    
    return img

df = pd.read_csv('raw_dimension_qas_mini.csv')
for i, row in df.iterrows():

    # Six views image   
    pdf_path = 'six_views.pdf'
    baseline_six_img = convert_pdf_to_images(pdf_path)

    baseline_six_img = crop_image(baseline_six_img, 200, 690, 1700, 90)
    baseline_six_img = rotate_image(baseline_six_img, -90)

    # Single image
    pdf_name = row['image_name'] + '.pdf'
    pdf_path = 'dimension_pdfs/' + pdf_name
    zoomed_im = convert_pdf_to_images(pdf_path)
    zoomed_im = crop_image(zoomed_im, 0, 0, 0, 1800)
    concat_im = concatenate_images(baseline_six_img, zoomed_im)

    # Draw line on image
    draw_img = draw_line_img(concat_im, zoomed_im.height, 5, 370)
    draw_img = draw_line_img(concat_im, concat_im.height - 570, 7, 360)

    # Overlay coordinate frame
    if row['view'] == "front":
        overlay = Image.open('front_coord.png')
        overlay = overlay.resize((overlay.width*6, overlay.height*6))
        draw_img.paste(overlay, (500, 500), overlay if overlay.mode == 'RGBA' else None)
        draw_img.show()
    
    # Get the question
    # TODO: finish prompting, decide how we want to organize the set of QAs where we ask about the rule with text included
    # versus asking about the rule without text included
    if row['dimension_system'] == "direct":
        question = prompt_preamble + 'The attached image shows two engineering drawings of our designed vehicle.'\
        ' The top vehicle shows two regions of our drawing, while the bottom '
    else:
        question = prompt_preamble + 'The attached image shows XX dimensions, three of which '
        
    answer = ""
    

