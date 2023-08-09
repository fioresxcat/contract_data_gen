from pdf2image import convert_from_path
from pathlib import Path
import os

dir = '/home/fiores/Desktop/VNG/VNG-DataGeneration/poc_training'
for pdf_fp in Path(dir).rglob('*.pdf'):
    images = convert_from_path(str(pdf_fp))
    for i in range(len(images)):
        out_fp = str(pdf_fp).replace('poc_training', 'poc_training_images').replace('.pdf', f'_{i}.jpg')
        if not Path(out_fp).parent.exists():
            os.makedirs(Path(out_fp).parent)
        images[i].save(out_fp)
    print(f'saved {pdf_fp} with {len(images)} images')