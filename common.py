import os
import unidecode
import numpy as np
import cv2
from PIL import Image, ImageDraw
import PIL.Image as Image
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageFilter as ImageFilter
from PIL import ImageEnhance
import string
import json
from random import randint
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import albumentations as A
import math
import random
from typing import Dict, Any
import pdb


font_scale = 32

normal = ImageFont.truetype(
    os.getcwd() + "/fonts/Arial/ARIAL.ttf", size=font_scale)

hsd = ImageFont.truetype(
    os.getcwd() + "/fonts/Arial/ARIAL.ttf", size=font_scale-20)
bold = ImageFont.truetype(
    os.getcwd() + "/fonts/Arial/ARIALBD.ttf", size=font_scale+randint(-5, 5))
italic = ImageFont.truetype(
    os.getcwd() + "/fonts/Arial/ARIALI.ttf", size=font_scale-10)
big_bold = ImageFont.truetype(
    os.getcwd() + "/fonts/Arial/ARIALBD.ttf", size=font_scale + 10)
Bold_Italic = ImageFont.truetype(
    os.getcwd() + "/fonts/Arial/ARIALBI.ttf", size=font_scale)
code_font = ImageFont.truetype(
    os.getcwd() + "/fonts/Arial/ARIAL.ttf", size=font_scale + 20)

roboto = ImageFont.truetype('fonts/Roboto-Bold.ttf', font_scale + randint(-5, 5))
inhoa_1 = np.random.choice([
    ImageFont.truetype(os.getcwd() + '/fonts/KGNoRegretsSolid.ttf',
                        font_scale+5),
    ImageFont.truetype(
        os.getcwd() + '/fonts/Hand Scribble Sketch Times.otf', font_scale+5)
])

bsx_font = ImageFont.truetype(
    os.getcwd() + '/fonts/Hand Scribble Sketch Times.otf', int(font_scale*2.5))
bsx_font_small = ImageFont.truetype(
    os.getcwd() + '/fonts/Hand Scribble Sketch Times.otf', font_scale+5)

inhoa_2 = np.random.choice([
    ImageFont.truetype(
        os.getcwd() + '/fonts/SourceSerifPro-Semibold.otf', font_scale+5),
    ImageFont.truetype(
        os.getcwd() + '/fonts/DroidSerif-Regular.ttf', font_scale+5),
    ImageFont.truetype(
        os.getcwd() + '/fonts/Times-New-Roman-Bold_44652.ttf', font_scale+5),
])

date2text = {
    '01': 'một',
    '02': 'hai',
    '03': 'ba',
    '04': 'bốn',
    '05': 'năm',
    '06': 'sáu',
    '07': 'bảy',
    '08': 'tám',
    '09': 'chín',
    '10': 'mười',
    '11': 'mười một',
    '12': 'mười hai',
    '13': 'mười ba',
    '14': 'mười bốn',
    '15': 'mười lăm',
    '16': 'mười sáu',
    '17': 'mười bảy',
    '18': 'mười tám',
    '19': 'mười chín',
    '20': 'hai mươi',
    '21': np.random.choice(['hai mươi mốt', 'hai mốt']),
    '22': np.random.choice(['hai mươi hai', 'hai hai']),
    '23': np.random.choice(['hai mươi ba', 'hai ba']),
    '24': np.random.choice(['hai mươi bốn', 'hai bốn']),
    '25': np.random.choice(['hai mươi lăm', 'hai lăm']),
    '26': np.random.choice(['hai mươi sáu', 'hai sáu']),
    '27': np.random.choice(['hai mươi bảy', 'hai bảy']),
    '28': np.random.choice(['hai mươi tám', 'hai tám']),
    '29': np.random.choice(['hai mươi chín', 'hai chín']),
    '30': 'ba mươi',
    '31': np.random.choice(['ba mươi mốt', 'ba mốt']),
}

digit2text = {
    '0': 'không',
    '1': 'một',
    '2': 'hai',
    '3': 'ba',
    '4': 'bốn',
    '5': 'năm',
    '6': 'sáu',
    '7': 'bảy',
    '8': 'tám',
    '9': 'chín',
}

common_words = [
    'chạy',
    'đi',
    'đá',
    'bơi',
    'đi bộ',
    'công chức',
    'công nhân',
    'bố',
    'mẹ',
    'con',
    'chồng',
    'vợ',
    'đàn ông',
    'phụ nữ',
    'người',
    'người ta',
    'người đàn ông',
    'người phụ nữ',
    'người mẹ',
    'người bố',
    'người con',
    'cháu',
    'ông',
    'bà',
    'lại',
    'bản chính',
    'bản gốc',
    'bản thân',
    'bản sao',
    'đi chơi',
    'chúc chích',
    'chúc mừng',
    'lần',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
]

def random_phrase(num_word):
    phrase = ''
    for _ in range(num_word):
        phrase += np.random.choice(common_words) + ' '
    return phrase

def year2text(year):
    assert len(year) == 4, "year must be 4 digits"
    assert year.isdigit(), "year must be digit"

    text = ''

    # year[0]
    if year[0] == '1':
        text += np.random.choice(['một ngàn', 'một nghìn'])
    elif year[0] == '2':
        text += np.random.choice(['hai ngàn', 'hai nghìn'])
    
    if year[1:] == '000':
        return text

    # year[1]
    if year[1] != '0':
        text += ' ' + digit2text[year[1]] + ' trăm'

    # year[2]
    if year[2] == '0':
        text += np.random.choice([' linh', ' lẻ'])
    elif year[2] == '1':
        text += ' mười'
    else:
        text += ' ' + digit2text[year[2]] + np.random.choice([' mươi', ''])
    
    # year[3]
    if year[3] == '0':
        if year[2] == '0':
            # remove last word from text
            text = ' '.join(text.split(' ')[:-1])
    elif year[3] == '1':
        if year[2] == '1' or (year[2]=='0' and year[1]=='0'):
            text += ' một'
        else:
            text += ' mốt'
    elif year[3] == '4':
        text += np.random.choice([' bốn', ' tư'])
    elif year[3] == '5':
        text += 'lăm'
    else:
        text += ' ' + digit2text[year[3]]


    return text

def dob2text(dob):
    date, month, year = dob.split('/')

    date_text = 'Ngày ' + date2text[date] + ', '
    month_text = 'tháng ' + date2text[month] + ', '
    year_text = 'năm ' + year2text(year)

    return date_text + month_text + year_text

def rand_ethnic():
    with open('ethnic_final.txt', 'r') as f:
        lines = f.readlines()
    return np.random.choice(lines)

def remove_accent(text):
    return unidecode.unidecode(text)
    
def randint(a, b):
    return np.random.randint(a, b)

def rand_normal(a, b):
    return np.random.normal(a, b)

def randink(bold=False, extra_bold=False):

    if bold:
        return tuple([int(np.random.normal(30, 10))] * 4)
    
    if extra_bold:
        return tuple([int(np.random.normal(15, 5))] * 4)
        
    return tuple([int(np.random.normal(50, 10))] * 4)

def randink_blue(bold=False):
    if bold:
        return (0, 153, 255)
    ls = [(26, 117, 255), (51, 133, 255), (77, 148, 255), \
                            (0, 92, 230), (51, 102, 255)]
                    
    return ls[randint(0, len(ls))]

def random_number(num_digit):
    number = ''
    for _ in range(num_digit):
        number += str(np.random.randint(0, 10))
    return str(number)

def random_character(num_char, upper='true', number='false'):
    if upper == 'true':
        if number == 'true':
            return ''.join(np.random.choice(list(string.ascii_letters + string.digits), num_char))
        return ''.join(np.random.choice(list(string.ascii_uppercase), num_char))
    elif upper == 'false':
        return ''.join(np.random.choice(list(string.ascii_lowercase), num_char))
    elif upper == 'all':
        special_chars = ['.', '/', ':', '{', '}', '|',  '&',  '(', ')', '-']
        return ''.join(np.random.choice(list(string.ascii_letters) + special_chars, num_char))
    
def rounded_img(img):
    img = Image.fromarray(img)
    w, h = img.size
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    offset = np.random.randint(5, 10)
    draw.rounded_rectangle((offset, offset, img.size[0]-offset, img.size[1]-offset), radius=np.random.randint(10, 20), fill=255)
    img.putalpha(mask)
    return np.array(img)

def ep_plastic(img):
    roi_img = Image.fromarray(img).convert('RGBA')
    roi_w, roi_h = roi_img.size
    pad = min(roi_w, roi_h) // 20
    val = (np.random.randint(200, 255), ) + (np.random.randint(200, 255), ) + (np.random.randint(200, 255), ) + (np.random.randint(80, 120), )
    roi_plastic = Image.new('RGBA', (roi_w+pad*2, roi_h+pad*2), val)
    roi_plastic.paste(roi_img, (pad, pad), roi_img)

    mask = Image.new('L', roi_plastic.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, roi_plastic.size[0], roi_plastic.size[1]), radius=np.random.randint(40, 60), fill=np.random.randint(60, 100))
    draw.rectangle((pad, pad, pad+roi_w, pad+roi_h), fill=255)
    roi_plastic.putalpha(mask)
    final_roi = np.array(roi_plastic)

    return final_roi

def vien_trang(img):
    pad_value = np.random.randint(200, 255)
    pad_size = min(img.shape[:2]) // 20
    padded_roi = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])
    
    return padded_roi

def random_vn_sentences():
    with open('common_vietnamese_words.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.split('-')[0][:-1] for line in lines]
    
    words = np.random.choice(lines, size=np.random.randint(8, 15), replace=False)
    # capitalize first letter in each word
    words = [word[0].upper() + word[1:] if np.random.rand() < 0.15 else word for word in words]
    return ' '.join(words)

def get_info(path="ds_cty.txt"):
    """
        company_names: list of all company name in the file ds_cty.txt
        names: list of ten cua nguoi dai dien cong ty 
    """
    with open(path, 'r', encoding='utf-8') as f:
        raws = f.read().split('\n')
    raws = [r for r in raws if r != '']
    names = [r.split(':')[-1] for r in raws[1::3]]
    addresses = [r[9:] for r in raws[2::3]]
    
    return names, addresses

def get_brand(path):
    with open(path, 'r') as f:
        brands = f.read().split('\n')
    
    # filter only keep element that has more than one character
    brands = [b for b in brands if len(b) > 1]
    return brands

def get_model_code(path):
    with open(path, 'r') as f:
        codes = f.read().split('\n')
    
    codes = [c for c in codes if not c.startswith('#') and len(c)>1]
    return codes

def correct_address(address):
    ls_remove = ['tỉnh', 'thành phố', 'quận', "huyện", "thị xã", 'đường', 'thị trấn', 'phường']
    address = address.lower()
    for r in ls_remove:
        address = address.replace(r, '')
    
    # title
    address = address.title()

    # remove 2 consecutive space
    address = ' '.join(address.split())
    
    address = address.replace(',,', ',')
    address = address.replace(', ,', ',')
    address = address.replace(' ,', ',')

    return address

def random_rotate(image , deg_range = 15 , border = (255,255,255)):
    d = np.random.randint(-deg_range,deg_range)
    h ,w = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), d, 1.0)
    # print(M)
    rotated = cv2.warpAffine(image, M, (w, h),borderValue = border ) 
    # rotated[rotated < 10 ] = 255
    return rotated 

def constrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val)/(max_val - min_val)  * 255

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 3
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
        
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def random_date(split='/'):
    # randomly generate a date of birth
    year = np.random.randint(1800, 2100)
    month = np.random.randint(1, 13)
    day = np.random.randint(1, 32)
    return str(day) + '/' + str(month) + '/' + str(year)

def rotate_bound(image, angle, pts=[[0.01, 0.01], [0.99, 0.01], [0.99, 0.99], [0.01, 0.99]]):
    """
        img: src image
        angle: rotate angle
        pts: coords of four corners of the real roi to paste on background (absolute value)
        roi (rectangular) = real roi (polygon) + zero padding
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image (shape của image sau khi xoay)
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # inverse matrix of simple rotation is reversed rotation.
    M_inv = cv2.getRotationMatrix2D((cX, cY), angle * (-1), 1)
    M_inv[0, 2] += (nW / 2) - cX
    M_inv[1, 2] += (nH / 2) - cY

    # points
    pts = np.array(pts)
    # add ones
    ones = np.ones(shape=(len(pts), 1))
    points_ones = np.hstack([pts, ones])

    # transform points: maxtrix xoay * (coords để paste img trên background)
    transformed_points = M_inv.dot(points_ones.T).T

    # return lại ảnh đã rotate
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255)), transformed_points
    
def random_capitalize(text, p):
    return np.random.choice([text.title(), text.upper(), text], p=p)

def random_space(text):
    """
        randomly replace a space in text with 0-5 space
    """
    space_idx = [idx for idx, c in enumerate(text) if c == ' ']
    if len(space_idx) > 0 and np.random.rand() < 0.25:
        idx2replace = np.random.choice(space_idx)
        num_space = np.random.randint(1, 6)
        text = text[:idx2replace] + ' ' * num_space + text[idx2replace+1:]
    return text

def split_text(text, factor_range):
    """
        cut text randomly at at least <factor> * text_length index
    """
    lowest = int(factor_range[0] * len(text))
    highest = int(factor_range[1] * len(text))
    split_indices = [idx for idx, c in enumerate(text) if highest > idx > lowest and c == ' ']
    if len(split_indices) == 0:
        return [text]
    else:
        idx = np.random.choice(split_indices)
        return [text[:idx], text[idx+1:]]

def get_last_length(text, font):
    return font.getsize(text[-1])[0]

def get_text_height(text, font=None):    
    max = 0
    for i in range(len(text)):
        if font.getsize(text[i])[1] > max:
            max = font.getsize(text[i])[1]
    
    return max

def get_num_char(text):
    # return number of alsphabets in text
    return len([c for c in text if c.isalpha() or c.isdigit()])

def widen_box(xmin, ymin, xmax, ymax, cut=True, size = None):     
    w, h = xmax - xmin, ymax - ymin
    # widen by 10% w
    xmin -= int(w * 0.05)
    xmax += int(w * 0.05)
    # widen by 10% h
    ymin -= int(h * 0.15)
    ymax += int(h * 0.15)

    if cut:
        if xmax > size[0]:
            xmax = size[0]
        if ymax > size[1]:
            ymax = size[1]

    # print(xmin , ymin , xmax , ymax)
    return xmin, ymin, xmax, ymax

# luu toa do cac box ra file json
def to_json(fp, fields, shape):
    h, w = shape[:2]
    json_dicts = {'shapes': [], 'imagePath': fp.split('/')[-1].replace('.json', '.jpg'),
                    'imageData': None, 'imageHeight': h, 'imageWidth': w}

    for field in fields:
        box = field['box']
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        tl = [x1, y1]
        tr = [x2, y2]
        br = [x3, y3]
        bl = [x4, y4]
        if tl[0] > w or tl[1] > h:
            continue

        coords = [tl, tr, br, bl]
        type = field["type"]

        json_dicts["shapes"].append(
            {'label': type, "text": field["text"], 'points': coords, 'shape_type': 'polygon', 'flags': {}})

    # print(json_path)
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(json_dicts, f)

def to_xml(xml_path, imgname, boxes, labels, shape):
    h, w = shape
    root = ET.Element('annotations')
    filename = ET.SubElement(root, 'filename')
    filename.text = imgname
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    width.text, height.text = str(w), str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = label
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin, ymin = ET.SubElement(bndbox, 'xmin'), ET.SubElement(bndbox, 'ymin')
        xmax, ymax = ET.SubElement(bndbox, 'xmax'), ET.SubElement(bndbox, 'ymax')
        xmin.text, ymin.text, xmax.text, ymax.text = [str(b) for b in box]
    ET.ElementTree(root).write(xml_path)


def mapping(boxes, position):
    '''
    Args: 
        boxes: List boxes of smaller module
        position: (x, y) position where to paste module ==> top-left
    Output: 
        list boxes in larger coordinate space
    Note that function not change order of box in list
    '''
    boxes = np.array(boxes) 
    new_boxes = np.copy(boxes)
    new_boxes[:, [0, 2]] = boxes[:, [0, 2]] + position[0]
    new_boxes[:, [1, 3]] = boxes[:, [1, 3]] + position[1]
    return new_boxes

def resize(new_shape, img, boxes):
    new_h, new_w = new_shape
    h, w = img.shape[:2]
    scale_x, scale_y = new_w / w, new_h / h
    new_img = cv2.resize(img, (new_w, new_h))
    if isinstance(boxes, list):
        new_boxes = [[x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y] for (x1, y1, x2, y2) in boxes]
    else: #numpy array
        new_boxes = np.copy(boxes)
        new_boxes[:, 0] = boxes[:, 0] * scale_x
        new_boxes[:, 1] = boxes[:, 1] * scale_y
        new_boxes = new_boxes.tolist()
    
    return new_img, new_boxes

def PIL_augment(image: Image):
    # Add noise
    if np.random.rand() < 0:
        noise = np.random.randint(50, 100)
        image = ImageEnhance.Brightness(image).enhance(noise / 100)
    # blur the image
    if np.random.rand() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 0.8)))
    # random brightness
    if np.random.rand() < 0.3:
        if np.random.rand() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(np.random.uniform(0.9, 1.1))
        else:
            image = ImageEnhance.Contrast(image).enhance(np.random.uniform(0.9, 1.1))
    
    return image

def erode_dilate(image: Image):
    # Create a structuring element
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    # kernel_size = np.random.choice([2, 3])
    kernel_size = 2
    # Erode the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    img = cv2.erode(img, kernel, iterations=1)
    # Dilate the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size-1, kernel_size-1))
    img = cv2.dilate(img, kernel, iterations=1)

    return Image.fromarray(img).convert('RGB')

def random_drop_black_pixel(image: Image, thresold=50):
    img = np.array(image)
    # create a mask of pixel value < 100
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # mask all positions that has pixel value < something < threshold
    mask = gray_img < np.random.rand(*gray_img.shape) * thresold
    # randomly set True to False in mask to lessen the effect
    mask = np.where(np.random.rand(*mask.shape) < 0.5, False, mask)
    # set the pixel value to white
    img[mask] = np.random.randint(100, 255)
    
    
    return Image.fromarray(img).convert('RGB')

class BaseAugmenter:
    def __init__(self, augmenter, size=None):
        self.size = size
        self.augmenter = augmenter
    
    def __call__(self, image, boxes):
        # assert len(boxes.shape) == 3
        keypoints = np.copy(boxes.reshape((-1, 2)))
        trans = self.augmenter(image=image.copy(), keypoints=keypoints)
        aug_img = trans['image']
        aug_boxes = np.reshape(trans['keypoints'], (-1, 4, 2))
        size = self.size if self.size is not None else image.shape[:2]
        aug_boxes = self.validate_boxes(aug_boxes, size)
        return aug_img, aug_boxes

    def validate_boxes(self, boxes, size):
        size_h, size_w = size
        new_boxes = []
        for box in boxes:
            # pdb.set_trace() 
            if np.sum(box[:, 0] > size_w) > 0 or np.sum(box[:, 0] < 0) > 0:
                continue
            if np.sum(box[:, 1] > size_h) > 0 or np.sum(box[:, 1] < 0) > 0:
                continue
            new_boxes.append(box)
        return np.array(new_boxes, dtype=np.int32)
    

class MySafeRotate(A.SafeRotate):
    def __init__(self, limit=3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None, mask_value=None, always_apply=False, p=0.5):
        super().__init__(limit, interpolation, border_mode, value, mask_value, always_apply, p)
        self.limit = limit

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        angle = self.limit

        image = params["image"]
        h, w = image.shape[:2]

        # https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
        image_center = (w / 2, h / 2)

        # Rotation Matrix
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        new_w = math.ceil(h * abs_sin + w * abs_cos)
        new_h = math.ceil(h * abs_cos + w * abs_sin)

        scale_x = w / new_w
        scale_y = h / new_h

        # Shift the image to create padding
        rotation_mat[0, 2] += new_w / 2 - image_center[0]
        rotation_mat[1, 2] += new_h / 2 - image_center[1]

        # Rescale to original size
        scale_mat = np.diag(np.ones(3))
        scale_mat[0, 0] *= scale_x
        scale_mat[1, 1] *= scale_y
        _tmp = np.diag(np.ones(3))
        _tmp[:2] = rotation_mat
        _tmp = scale_mat @ _tmp
        rotation_mat = _tmp[:2]

        return {"matrix": rotation_mat, "angle": angle, "scale_x": scale_x, "scale_y": scale_y}


class RandomRotate:
    def __init__(self, limit=3):
        augmenter = A.Compose([MySafeRotate(limit, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], p=1.0)],
                              keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        self.base_augmenter = BaseAugmenter(augmenter, None)
    def __call__(self, image, boxes):
        return self.base_augmenter(image, boxes)
    
def rotate_img_after_gen(img, fields):
    boxes = []
    for field in fields:
        x1, y1, x2, y2, x3, y3, x4, y4 = field['box']
        boxes.extend([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    rotate_angle = np.random.randint(-3, 4)
    rotater = RandomRotate(limit=rotate_angle)
    rotated_img, rotated_boxes = rotater(img, np.array(boxes))
    for i, field in enumerate(fields):
        fields[i]['box'] = [coord for pt in rotated_boxes[i] for coord in pt]
    
    return rotated_img, fields, rotate_angle

def augment_scan(dir):
    """
        make all img in dir look scanned
        need Image Magick to be installed
    """
    import subprocess
    import os

    for fn in os.listdir(dir):
        if not fn.endswith('.jpg'):
            continue
        fp = os.path.join(dir, fn)
        command = f'convert -density 150 {fp} -colorspace gray -linear-stretch 3.5%x10% -blur 0x0.5 -attenuate 0.25 +noise Gaussian {fp}'
        num_repeat = np.random.choice([1, 2])
        for _ in range(num_repeat):
            res = subprocess.run(command, shell=True)
        print(f'done {fp}')

def double_case_prob(prob = 0.2):
    if np.random.random() < 0.2:
        return 1
    else:
        return 0