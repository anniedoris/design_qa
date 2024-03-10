from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import pandas as pd
import sys
sys.path.append("../..")
from common_prompts import prompt_preamble

def convert_pdf_to_images(pdf_path, index):
    """
    Convert single page PDF to an image.

    Parameters:
    - pdf_path: Path to the PDF file.
    """
    # Convert PDF to a list of images
    images = convert_from_path(pdf_path, dpi=600)
    
    return images[index]

def convert_single_pdf_to_images(pdf_path):
    """
    Convert single page PDF to an image.

    Parameters:
    - pdf_path: Path to the PDF file.
    """
    # Convert PDF to a list of images
    images = convert_from_path(pdf_path, dpi=600)
    
    for i, image in enumerate(images):
        pass
    
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

def crop_cad_image(img, left, top, right, bottom):
    """
    Crop the image from the left, top, right, and bottom

    Parameters:
    - image_path: Path to the image file.
    - left: Amount to crop from the left.
    - top: Amount to crop from the top.
    - right: Amount to crop from the right.
    - bottom: Amount to crop from the bottom.
    """
        
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
    
    return rotated_img

def rotate_image(img, rotation_angle):
    rotated_img = img.rotate(rotation_angle, expand=True)
    return rotated_img

def concatenate_images(base_img, zoomed_img):
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

df = pd.read_csv('raw_dimension_qas.csv')
# df = pd.read_csv('raw_T.8.2.4.csv')
detailed_context = True # sets whether the hint images are appended (True) or not (False)

qa = []
for i, row in df.iterrows():
    
    print("RULE")
    print(row['rule_tested'])
    if detailed_context:
        context_img = convert_pdf_to_images('rule_evaluation_definition_detailed_context.pdf', int(row['context_im_detailed']))
    else:
        context_img = convert_pdf_to_images('rule_evaluation_definition_context.pdf', int(row['context_im']))
    cropped_context_img = crop_cad_image(context_img, left=550, top=600, right=2275, bottom=0)

    # Single image
    pdf_name = row['image_name'] + '.pdf'
    pdf_path = 'dimension_pdfs/' + pdf_name
    zoomed_im = convert_single_pdf_to_images(pdf_path)
    zoomed_im = crop_image(zoomed_im, 0, 0, 0, 1800)
    concat_im = concatenate_images(cropped_context_img, zoomed_im)

    # Draw line on drawing image
    draw_img = draw_line_img(concat_im, zoomed_im.height, 5, 370)

    # Overlay coordinate frame on the drawing
    overlay = Image.open('coord_orientations/' + row['view'] + '_coord.png')
    overlay = overlay.resize((overlay.width*6, overlay.height*6))
    draw_img.paste(overlay, (draw_img.width - 1600, 700), overlay if overlay.mode == 'RGBA' else None)
    # overlay = overlay.resize((overlay.width*4, overlay.height*4)) # for raw_T.8.2.4 so coord doesn't overlap
    # draw_img.paste(overlay, (draw_img.width - 1200, 700), overlay if overlay.mode == 'RGBA' else None) # for raw_T.8.2.4 so coord doesn't overlap
    draw_img.show()
    
    rule_num = row['rule_tested']
    answer = row['complies']
    cad_model = row['cad_model']
    additional_info = str(row['additional_info'])
    if "nan" in additional_info:
        additional_info = ""
    
    if len(additional_info) > 1:
        additional_info += " "
    
    additional_info_context = ""
    if detailed_context:
        additional_info_context = str(row['additional_info_context'])
        
        if "nan" in additional_info_context:
            additional_info_context = ""
            
        if len(additional_info_context) > 1:
            additional_info_context += " "
    
    prompt_1 = f"Also attached is an image that shows an engineering drawing of the {cad_model} on the top accompanied by six CAD views of the {cad_model} on"\
        " the bottom. The six CAD views each feature a different orientation of our design, so that 3D information about our design can be inferred."\
        " " + additional_info_context + "The CAD views are provided to contextualize the engineering drawing, which has the same orientation as one of the six CAD views. All units displayed"\
        " in the engineering drawing have units of mm. " + additional_info + "Based on the engineering drawing, does our design comply with rule " + rule_num + " specified in the FSAE rule document?"
    prompt_2 = f" First provide an explanation for your answer (begin it with 'Explanation:'). Then provide just a yes/no answer"\
        " (begin it with 'Answer:') that summarizes your response."
        
    # For direct dimensioning questions
    if row['dimension_system'] == "direct":
        question = prompt_preamble + prompt_1 + \
        " Only use dimensions explicitly shown in the engineering drawing to answer the question. If a dimension is not explicitly shown, you can assume that it"\
        " complies with the rules." + prompt_2
        
        if answer == "yes":
            output_im_name = row['rule_tested'] + "a" + '.jpg'
        if answer == "no":
            output_im_name = row['rule_tested'] + "b" + '.jpg'
    
    # For scale bar questions 
    else:
        question = prompt_preamble + prompt_1 + \
        " To answer the question, use the scale bar shown at the top of the engineering drawing to compute necessary dimensions in the drawing. " +\
        prompt_2
        
        output_im_name = row['rule_tested'] + "c" + '.jpg'
    
    # Save the generated image
    if detailed_context:
        draw_img.save("../../../dataset/rule_compliance/rule_dimension_qa/detailed_context/" + output_im_name)
    else:
        draw_img.save("../../../dataset/rule_compliance/rule_dimension_qa/context/" + output_im_name)
    
    dimension_type = row['dimension_system']
    explanation = row['explanation']
    qa.append([question, answer, output_im_name, dimension_type, explanation])

if detailed_context:    
    pd.DataFrame(qa, columns=['question', 'ground_truth', 'image', 'dimension_type', 'explanation']).to_csv(
        "../../../dataset/rule_compliance/rule_dimension_qa/detailed_context/rule_dimension_qa_detailed_context.csv", index=False, encoding='utf-8')
else:
    pd.DataFrame(qa, columns=['question', 'ground_truth', 'image', 'dimension_type', 'explanation']).to_csv(
        "../../../dataset/rule_compliance/rule_dimension_qa/context/rule_dimension_qa_context.csv", index=False, encoding='utf-8')
    
    

