import pdfplumber
from tqdm import tqdm

all_txt = ''
with pdfplumber.open("../../dataset/docs/FSAE_Rules_2024_V1.pdf") as pdf:
    number_of_pages = len(pdf.pages)
    for page_num in tqdm(range(number_of_pages)):
        all_txt += pdf.pages[page_num].extract_text() + "\n"

with open("../../dataset/docs/rules_pdfplumber1.txt", "w", encoding="utf-8") as f:
    f.write(all_txt)
