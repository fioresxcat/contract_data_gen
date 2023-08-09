import os
import numpy as np
from pathlib import Path
import shutil
import json

# dir = 'resources'
# fn = 'raw_foreign_address.txt'
# out_dir = 'resources'
# with open(os.path.join(dir, fn)) as f:
#     data = f.readlines()
#     data = [line for line in data if len(line.split()) >= 2]
# print(len(data))

# data = np.random.choice(data, int(1e5), replace=False)
# with open(os.path.join(out_dir, fn), 'w') as f:
#     for line in data:
#         f.write(line)

# dir = os.path.join(os.getcwd(), 'contract/result')
# ls_img = [str(img_fp) for img_fp in Path(dir).rglob('*.jpg')]
# ls_img2keep = np.random.choice(ls_img, int(500), replace=False)
# ls_img2drop = list(set(ls_img) - set(ls_img2keep))
# cnt = 0
# for img_fp in ls_img2drop:
#     os.remove(img_fp)
#     cnt += 1
# print(f'removed {cnt} images')

# dir = os.path.join(os.getcwd(), 'contract/result')
# for json_fp in Path(dir).rglob('*.json'):
#     jpg_fp = json_fp.with_suffix('.jpg')
#     if not jpg_fp.exists():
#         xml_fp = json_fp.with_suffix('.xml')
#         os.remove(json_fp)
#         os.remove(xml_fp)

# root_dir = '/data2/tungtx2/VNG-DataGeneration/contract/nho_doi_at_thanh_marker_result_updated_position_26032023'
# ls_jpg = [jpg_fp for jpg_fp in Path(root_dir).rglob('*.jpg')]
# ls_jpg = np.random.choice(ls_jpg, 500)
# ls_xml = [fp.with_suffix('.xml') for fp in ls_jpg]
# ls_json = [fp.with_suffix('.json') for fp in ls_jpg]
# for fp in Path(root_dir).rglob('*'):
#     if fp not in ls_jpg and fp not in ls_xml and fp not in ls_json:
#         os.remove(fp)
# with open('/data2/tungtx2/VNG-DataGeneration/resources/all_position.txt') as f:
#     lines = f.readlines()

# lines = [line for line in lines if 'liason' not in line.lower()]
# with open('/data2/tungtx2/VNG-DataGeneration/resources/all_position_1.txt', 'w') as f:
#     for line in lines:
#         f.write(line)