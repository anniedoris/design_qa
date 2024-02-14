from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path, output_folder):
    """
    Convert each page of a PDF to an image.

    Parameters:
    - pdf_path: Path to the PDF file.
    - output_folder: Folder where the output images will be saved.
    """
    # Convert PDF to a list of images
    images = convert_from_path(pdf_path, dpi=600)

    # Save each page as an image
    for i, image in enumerate(images):
        image_path = f"{output_folder}/{i+1}.jpg"
        image.save(image_path, 'JPEG')
        print(f"Saved {image_path}")

# Example usage
pdf_path = 'rule_comprehension_presence.pdf'
output_folder = 'raw_presence_images'
convert_pdf_to_images(pdf_path, output_folder)
