import sys
import os
sys.path.append(os.getcwd())
from sub_modules.sub_modules import *
import numpy as np
from party import Party
from bank import Bank
from common import *
import pdb
import argparse

def blend(module, background, position):
    '''
    Args: 
        module: A Module
        background: an image, numpy array
    '''
    x, y = position
    hm, wm = module.get_shape()
    cx, cy = x + wm//2, y + hm//2
    img = cv2.seamlessClone(module.canvas, background, mask=None, p=(cx, cy), flags=cv2.MONOCHROME_TRANSFER)

    # Paste submodule and correct the box coordinate
    fields = module.get_fields()
    boxes = [field['box'] for field in fields]
    boxes = mapping(boxes, position)
    for i in range(len(boxes)):
        module.fields[i]['box'] = boxes[i]

    block_box = [x, y, x+wm, y+hm]
    return img, block_box

def blend_white(img, position, blank_shape):
    '''
    Args: 
        module: A Module
        background: an image, numpy array
    '''
    x, y = position
    h, w = blank_shape
    cx, cy = x + w//2, y + h//2
    # img = cv2.seamlessClone(blank, img, mask=None, p=(cx, cy), flags=cv2.MONOCHROME_TRANSFER)

    idxs = np.where(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) > 200)
    color = np.mean(img[idxs[0], idxs[1]], 0)
    color = [int(c) for c in color]
    blank = np.full(blank_shape + (3,), color, dtype='uint8')

    # pil blend
    img = Image.fromarray(img).convert('RGBA')
    blank_img = Image.fromarray(blank).convert('RGBA')
    background_part = img.crop((x, y, x+w, y+h))
    blended_part = Image.blend(background_part, blank_img, alpha=1)
    img.paste(blended_part, (x, y, x+w, y+h))
    img = np.array(img.convert('RGB'))

    return img

class ImageGen:
    def __init__(self, background_path):
        self.background_path = background_path
        self.names = [name for name in os.listdir(self.background_path)[:20] if name.endswith('.jpg')]
        
    def reset(self):
        self.modules = []
                
    def get_random_background(self):
        name = np.random.choice(self.names)
        background = {
            'img': cv2.imread(os.path.join(self.background_path, name)),
            'line_points': np.load(os.path.join(self.background_path, name[:-4] + '_lp_.jpg.npy')),
            'white_backgrounds': np.load(os.path.join(self.background_path, name[:-4] + '_bg_.jpg.npy')),
            'font_size': int(np.load(os.path.join(self.background_path, name[:-4] + '_fs_.jpg.npy')))
        }
        return background
    
    def gen_background(self, modules):
        # get sum height of all block
        sum_h = sum([module.get_shape()[0] for module in modules])
        # get information of background
        background = self.get_random_background()        
        bg_h, bg_w, _ = background['img'].shape
        # random position
        start_h = 0
        list_paste_points = []
        for module in modules:
            block_h, block_w = module.get_shape()
            mark_h = randint(start_h, randint(start_h, bg_h - sum_h))
            # find last line < mark_h
            start_line = [np.array([0, mark_h]), np.array([bg_w, mark_h])]
            for point1, point2 in background['line_points']:
                x1, y1 = point1
                if y1 >= mark_h:
                    break
                if y1 < start_h:
                    continue
                start_line = [point1, point2]
            # find first line > mark_h + block_h
            end_line = [np.array([0, mark_h + block_h]), np.array([bg_w, mark_h + block_h])]
            for point1, point2 in list(reversed(background['line_points'])):
                x1, y1 = point1
                if y1 < mark_h + block_h:
                    break
                end_line = [point1, point2]
            remove_points = np.array([start_line[0], start_line[1], end_line[1], end_line[0]])
            x1, y1 = int(np.min(remove_points[..., 0])), int(np.min(remove_points[..., 1]))
            x2, y2 = int(np.max(remove_points[..., 0])), int(np.max(remove_points[..., 1]))
            
            # paste white part to background
            background['img'] = blend_white(background['img'],  (0, y1), (y2-y1, bg_w))
            # choose paste points for block
            position_x = randint(0, bg_w - block_w - 1)
            position_y = y1 + randint(0, y2-y1-block_h)
           
            list_paste_points.append([position_x, position_y])
            start_h = y2 
            sum_h -= block_h
        return background, list_paste_points


    def gen_ordered_background(self, modules, party_order_type):
        party_modules = [module for module in modules if module.__class__.__name__ == 'Party']
        bank_modules = [module for module in modules if module.__class__.__name__ == 'Bank']
        background = self.get_random_background()        
        bg_h, bg_w, _ = background['img'].shape

        # pary modules: 2 type
        list_paste_points = []
        list_return_modules = []
        start_h = bg_h // 10  # vi tri dau tien trong background de paste cac sub module

        if party_order_type == 'left|right':
            max_w = max([module.get_shape()[1] for module in party_modules])
            if max_w >= bg_w//2 * 1.3:  # neu module la qua to, thi ta se lai paste lan luot vay
                party_order_type == 'up|down'

        if party_order_type == 'up|down':
            # get sum height of all block
            sum_h = sum([module.get_shape()[0] for module in modules])
            # random position
            position_x = None
            for module in party_modules:
                block_h, block_w = module.get_shape()
                mark_h = randint(start_h+20, start_h + 40)  # 2 module lien tip chi cach nhau mot chut
                # get region to paste
                x1, y1, x2, y2 = self.find_region_to_paste(start_h, mark_h, block_h, background)
                # paste white part to background
                background['img'] = blend_white(background['img'],  (0, y1), (y2-y1, bg_w))
                # choose paste points for block
                position_x = randint(20, bg_w - block_w - 1) if position_x is None else position_x + randint(-10, 10)
                position_y = y1 + randint(0, y2-y1-block_h)
                # update stuff
                list_paste_points.append([position_x, position_y])
                start_h = y2   # cap nhat start_h cho module tiep theo
                sum_h -= block_h

                list_return_modules.append(module)

        elif party_order_type == 'left|right':
            # get max height of module
            max_h = max([module.get_shape()[0] for module in modules])
            mark_h = randint(start_h, start_h + randint(10, 50)) 
            # get region to paste
            x1, y1, x2, y2 = self.find_region_to_paste(start_h, mark_h, max_h, background)
            start_h = y2     # cap nhat start_h cho module tiep theo
            # paste white part to background
            background['img'] = blend_white(background['img'],  (0, y1), (y2-y1, bg_w))

            for i, module in enumerate(party_modules):
                block_h, block_w = module.get_shape()
                if block_w > bg_w / 2:
                    new_w = int(bg_w / 2) - 20
                    ratio = block_w / new_w
                    new_h = int(block_h / ratio)
                    module.resize((new_h, new_w))
                    block_h, block_w = module.get_shape()
                    
                # choose paste points for block
                if i == 0: # first party
                    position_x = randint(20, bg_w//2 - block_w - 1)
                else:  # second party
                    position_x = randint(bg_w//2+10, bg_w - block_w - 1)
                position_y = y1 + randint(0, y2-y1-block_h)
                list_paste_points.append([position_x, position_y])
                list_return_modules.append(module)

        # bank modules
        if len(bank_modules) > 0:
            bank_module = bank_modules[0]
            block_h, block_w = bank_module.get_shape()
            # get max height of module
            mark_h = randint(max(start_h, bg_h * 2//3), bg_h - block_h - 1)  # random vi tri cua bank module trong phan cuoi cua background
            # get region to paste
            x1, y1, x2, y2 = self.find_region_to_paste(start_h, mark_h, block_h, background)
            start_h = y2   # cap nhat start_h cho module tiep theo
            # paste white part to background
            background['img'] = blend_white(background['img'],  (0, y1), (y2-y1, bg_w))

            # choose paste points for block
            position_x = randint(0, bg_w - block_w - 1)
            position_y = y1 + randint(0, y2-y1-block_h)
            list_paste_points.append([position_x, position_y])
            list_return_modules.append(bank_module)

        return background, list_paste_points, list_return_modules
    

    def gen_ordered_background_aSon(self, modules, party_order_type):
        background = self.get_random_background()        
        bg_h, bg_w, _ = background['img'].shape

        # pary modules: 2 type
        list_paste_points = []
        list_return_modules = []
        start_h = 6 * bg_h // 10  # vi tri dau tien trong background de paste cac sub module

        if party_order_type == 'bank|bank':
            # get sum height of all block
            sum_h = sum([module.get_shape()[0] for module in modules])
            # random position
            position_x = None
            for module in modules:
                block_h, block_w = module.get_shape()
                mark_h = randint(start_h+5, start_h + 6)  # 2 module lien tip chi cach nhau mot chut
                # get region to paste
                x1, y1, x2, y2 = self.find_region_to_paste(start_h, mark_h, block_h, background)
                # paste white part to background
                background['img'] = blend_white(background['img'],  (0, y1), (y2-y1, bg_w))
                # choose paste points for block
                position_x = randint(20, bg_w - block_w - 1) if position_x is None else position_x + randint(-10, 10)
                position_y = y1 + randint(0, y2-y1-block_h)
                # update stuff
                list_paste_points.append([position_x, position_y])
                start_h = y2   # cap nhat start_h cho module tiep theo
                sum_h -= block_h

                list_return_modules.append(module)

        elif party_order_type == 'party|bank':
            # get max height of module
            max_h = max([module.get_shape()[0] for module in modules])
            mark_h = randint(start_h, start_h + randint(10, 70)) 
            # get region to paste
            x1, y1, x2, y2 = self.find_region_to_paste(start_h, mark_h, max_h, background)
            start_h = y2     # cap nhat start_h cho module tiep theo
            # paste white part to background
            background['img'] = blend_white(background['img'],  (0, y1), (y2-y1, bg_w))

            for i, module in enumerate(modules):
                block_h, block_w = module.get_shape()
                if block_w > bg_w / 2:
                    new_w = int(bg_w / 2) - 20
                    ratio = block_w / new_w
                    new_h = int(block_h / ratio)
                    module.resize((new_h, new_w))
                    block_h, block_w = module.get_shape()
                    
                # choose paste points for block
                if i == 0: # first party
                    position_x = randint(20, bg_w//2 - block_w )
                else:  # second party
                    position_x = randint(bg_w//2+10, bg_w - block_w - 1)
                position_y = y1 + randint(0, y2-y1-block_h)
                list_paste_points.append([position_x, position_y])
                list_return_modules.append(module)

        # # bank modules
        # if len(bank_modules) > 0:
        #     bank_module = bank_modules[0]
        #     block_h, block_w = bank_module.get_shape()
        #     # get max height of module
        #     mark_h = randint(max(start_h, bg_h * 2//3), bg_h - block_h - 1)  # random vi tri cua bank module trong phan cuoi cua background
        #     # get region to paste
        #     x1, y1, x2, y2 = self.find_region_to_paste(start_h, mark_h, block_h, background)
        #     start_h = y2   # cap nhat start_h cho module tiep theo
        #     # paste white part to background
        #     background['img'] = blend_white(background['img'],  (0, y1), (y2-y1, bg_w))

        #     # choose paste points for block
        #     position_x = randint(0, bg_w - block_w - 1)
        #     position_y = y1 + randint(0, y2-y1-block_h)
        #     list_paste_points.append([position_x, position_y])
        #     list_return_modules.append(bank_module)

        return background, list_paste_points, list_return_modules
    

    
    def find_region_to_paste(self, start_h, mark_h, block_h, background):
        bg_h, bg_w, _ = background['img'].shape
        # find last line < mark_h
        start_line = [np.array([0, mark_h]), np.array([bg_w, mark_h])]
        for point1, point2 in background['line_points']:
            x1, y1 = point1
            if y1 >= mark_h:
                break
            if y1 < start_h:
                continue
            start_line = [point1, point2]
        # find first line > mark_h + block_h
        end_line = [np.array([0, mark_h + block_h]), np.array([bg_w, mark_h + block_h])]
        for point1, point2 in list(reversed(background['line_points'])):
            x1, y1 = point1
            if y1 < mark_h + block_h:
                break
            end_line = [point1, point2]
        remove_points = np.array([start_line[0], start_line[1], end_line[1], end_line[0]])
        x1, y1 = int(np.min(remove_points[..., 0])), int(np.min(remove_points[..., 1]))
        x2, y2 = int(np.max(remove_points[..., 0])), int(np.max(remove_points[..., 1]))

        return x1, y1, x2, y2


    def gen_image(self, modules, party_order_type):
        # get background image
        background, list_paste_points, modules = self.gen_ordered_background_aSon(modules, party_order_type)
        # paste block_image into background image
        for module, paste_points in zip(modules, list_paste_points):
            background['img'], block_box = blend(module, background['img'], paste_points)
            self.modules.append({'box':block_box, 'module':module})
        return background['img']
    
    def get_fields(self):
        fields = []
        for module in self.modules:
            fields += list(module.get_fields())
        return fields

    
    
def init_bank(font_size):
    font = Font(font_scale=font_size)
    font_normal, font_bold, font_italic = font.get_font('normal'), font.get_font('bold'), font.get_font('italic')
    def rand_font():
        return np.random.choice([font_normal, font_bold, font_italic])

    bank_name = BankName(rand_font(), rand_font(),marker_prob=0.7, down_prob=0)
    bank_name.random_capitalize_prob = {
        'marker': [0.5, 0.5, 0],  
        'content': [0.2, 0.8, 0] 
    }
    bank_name()
    
    bank_address = Bank_Address(rand_font(), rand_font(),marker_prob=0.7, down_prob=0.2)
    bank_address()

    company_address = Company_Address(rand_font(), rand_font(),marker_prob=0.7, down_prob=0.2)
    company_address.module_order_type = 'up|down'
    company_address.random_capitalize_prob = {
        'marker': [0.4, 0.5, 0.1],
        'content': [0.4, 0.5, 0.1]
    }
    company_address()


    account_number = AccountNumber(rand_font(), rand_font(), marker_prob=1, down_prob=0)
    account_number()

    account_name = AccountName(rand_font(), rand_font(), marker_prob=1, down_prob=0.2)
    account_name.random_capitalize_prob = {
        'marker': [0.6, 0.4, 0],  
        'content': [0.2, 0.8, 0]
    }
    account_name()

    swift_code = SwiftCode(rand_font(), rand_font(), marker_prob=1, down_prob=0)
    swift_code.random_capitalize_prob = {
        'marker': [0.6, 0.4, 0],  
        'content': [0, 1, 0]  
    }
    swift_code()

    bank = Bank()
    bank(bank_name, bank_address, company_address, account_number, account_name, swift_code)
    return bank

if __name__ == '__main__':
    from time import time

    parser = argparse.ArgumentParser(
                    prog='ContractGen',
                    description='',
                    epilog='')
    
    parser.add_argument('--bg_path', type = str, default='/data2/users/common/corpora/vision/temp_data/block_to_image/backgrounds',
                        help='Backgound path')
    parser.add_argument('--count', type = int, default=400,
                        help='Number of generated images')
    args = parser.parse_args()

    gener = ImageGen(args.bg_path)
    save_path = os.getcwd() + "/contract/nho_doi_at_thanh_marker_result_06042023_aSon_party_bank/"
    os.makedirs(save_path, exist_ok=True)
    view = False
    save = True
    
    # field['box'] = [x1, y1, x2, y2, x3, y3, x4, y4]
    # block = [xmin, ymin, xmax, ymax]
    img_cnt = 0
    while img_cnt < 100:
        try:
            gener.reset()
            background = gener.get_random_background()
            font_size = background['font_size']

            order_type = np.random.choice(['bank|bank', 'party|bank'], p=[0, 1])
            modules = []
            if order_type == 'bank|bank':
                bank = init_bank(font_size)
                modules.append(bank)
                bank = init_bank(font_size)
                modules.append(bank)
            elif order_type == 'party|bank':
                partyA = Party(skip_prob = 0.1, down_prob = 0.6, bank_prob = 0.6, font_size=font_size)()
                bank = init_bank(font_size)
                modules.append(partyA)
                modules.append(bank)

            img = gener.gen_image(modules, party_order_type=order_type)
            fields = []
            blocks = []
            for module in gener.modules:
                blocks.append(module['box'])
                fields += module['module'].get_fields()
            # convert field from [x1, y1, x2, y2] to [x1, y1, x2, y2, x3, y3, x4, y4]
            for i, field in enumerate(fields):
                xmin, ymin, xmax, ymax = field['box']
                fields[i]['box'] = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]

            # # random rotate
            # is_rotate = np.random.random() < 0.5
            # if is_rotate:
            #     img, fields, rotate_angle = rotate_img_after_gen(img, fields)

            if view:
                for block in blocks:
                    x1, y1, x2, y2 = block
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), thickness = 1)
                
                for field in fields:
                    x1, y1, x2, y2, x3, y3, x4, y4 = field['box']
                    img = cv2.rectangle(img, (x1, y1), (x3, y3), (0, 255, 0), thickness = 1)
                cv2.imshow('img', img)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    exit(-1)

            if save:
                prefix = f'fake_{np.random.randint(0, 10000)}'

                ## Convert to int type for dump json
                for field in fields:
                    field['box'] = [int(p) for p in field['box']]

                to_json(os.path.join(save_path, prefix + '.json'), fields, img.shape[:2])
                to_xml(os.path.join(save_path, prefix + '.xml'), prefix+'.jpg', blocks, ['partyA', 'partyB', 'bank'], img.shape[:2])
                cv2.imwrite(os.path.join(save_path, prefix + '.jpg'), img)

            img_cnt += 1
            print(img_cnt)
        except Exception as e:
            # raise e     
            print(e)
            continue
            pass
