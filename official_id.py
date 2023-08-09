from calendar import c
from curses import raw
from hashlib import new
from os import remove
from re import L
import PIL.Image as Image
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageFilter as ImageFilter
from PIL import ImageEnhance
import numpy as np
import glob
import time
import pandas as pd
import json
import numpy as np
import cv2
import string
import xml.etree.ElementTree as ET
from common import *
import os

labels_list = []


class OfficialID:
    def __init__(self, dst):
        self.dst = dst
        # self.img_path = "fake_{}.jpg".format(np.random.randint(665527))
        self.img_name = "fake_test.jpg"

        # self.font_scale = np.random.randint(25, 26)
        # self.original_font_scale = self.font_scale

        # self.left_margin = randint(65, 75)
        # self.line_height, self.original_line_height = [randint(55, 57)] * 2

        self.img_width = 1200
        self.img_height = 1980 # ratio w/h = 1.6

        # bg_path = np.random.choice(glob.glob("bg/cmca/*"))
        # self.image = Image.open(bg_path).resize([self.img_width, self.img_height])
        # self.draw = ImageDraw.Draw(self.image)
        # self.cursor = [self.left_margin, randint(65, 75)]  # con trỏ đầu dòng
        # self.line = 0

        # define cac loai font
        # self.normal = ImageFont.truetype(
        #     "fonts/time_new_roman.ttf", size=self.font_scale)
        # self.big = ImageFont.truetype(
        #     "fonts/time_new_roman.ttf", size=self.font_scale+5)
        # self.small = ImageFont.truetype(
        #     "fonts/time_new_roman.ttf", size=self.font_scale-2)
        # self.extra_big = ImageFont.truetype(
        #     "fonts/time_new_roman.ttf", size=self.font_scale+10)

        # self.bold = ImageFont.truetype(
        #     "fonts/Times-New-Roman-Bold_44652.ttf", size=self.font_scale)
        # self.italic = ImageFont.truetype(
        #     "fonts/Times-New-Roman-Italic_44665.ttf", size=self.font_scale)
        # self.bold_italic = ImageFont.truetype(
        #     'fonts/Times-New-Roman-Bold-Italic_44651.ttf', self.font_scale+5)
        # self.big_bold = ImageFont.truetype( 
        #     "fonts/Times-New-Roman-Bold_44652.ttf", size=self.font_scale + 3)
        # self.extra_big_bold = ImageFont.truetype( \
        #     "fonts/Times-New-Roman-Bold_44652.ttf", size=self.font_scale + 25)
        # self.extra_small = ImageFont.truetype(
        #     "fonts/time_new_roman.ttf", size=self.font_scale-7)

        # font_path = np.random.choice(glob.glob("fonts/viet-tay/*"))
        # self.viet_tay = ImageFont.truetype(font_path, int(self.font_scale*2))

        # self.viet_tay_ink = np.random.choice([randink_blue(), randink()])

        

        # # chu ki con dau 
        # self.stamp_ink = (255, 51, 51)
        
        # chuki_condau_font_1_path = 'fonts/VNI/UTM Zirkon.ttf'
        # chuki_condau_font_2_path = 'fonts/Manrope-Medium.ttf'
        
        # self.chuki_condau_font_1 = ImageFont.truetype(chuki_condau_font_1_path, int(self.font_scale*2.5))
        # self.chuki_condau_font_2 = ImageFont.truetype(chuki_condau_font_2_path, int(self.font_scale*1.2))



        self.info = pd.read_csv("personal_info/eID_1000_front_v2.0.11 - eID_1000_front_v2.0.11.csv")

        n = len(self.info)-1
        # get third column of self.info as a list
        self.info.iloc[:, 2].tolist()
        self.province = eval(self.info.loc[np.random.randint(n)][9])["province"]
        print(self.province)
        assert len(self.province) > 0
        self.province_type = eval(self.info.loc[np.random.randint(n)][9])["province_type"]
        print('province type: ', self.province_type)
        assert(len(self.province_type) > 0)
        self.district = eval(self.info.loc[np.random.randint(n)][9])["district"]
        print('district: ', self.district)
        assert(len(self.district) > 0)
        self.district_type = eval(self.info.loc[np.random.randint(n)][9])["district_type"]
        print('district_type: ', self.district_type)
        assert(len(self.district_type) > 0)
        self.ward = eval(self.info.loc[np.random.randint(n)][9])["ward"]
        print('ward: ', self.ward)
        assert(len(self.ward) > 0)
        self.ward_type = eval(self.info.loc[np.random.randint(n)][9])["ward_type"]
        print('ward_type: ', self.ward_type)
        assert(len(self.ward_type) > 0)
        self.id = self.info.loc[np.random.randint(n)][1]
        print('id: ', self.id)
        assert(len(self.id) > 0)

        self.names, self.usual_addresses = get_info()  # địa chỉ thường trú
        self.names.extend(self.info.iloc[:, 2].tolist()) # all the random names
        self.usual_addresses.extend(self.info.iloc[:, 10].tolist()) # all the random địa chỉ thường trú

        self.hometown = self.info.iloc[:, 8].tolist()  # quê quán
        self.characters = list(string.ascii_uppercase + string.digits)  # all the characters possible


        self.blocks = []

        self.block_types = [
            'block_poi',
            'block_template',
            'block_info',
            'block_staff',
            'block_sign'
        ]

        self.fields = []
        # moi phan tu la 1 dictionary co dang
        # {
        #     'xmin': 0,
        #     'ymin': 0,
        #     'xmax': 0,
        #     'ymax': 0,
        #     'text': 'text',
        # }

        # block related
        # ko can quan tam
        self.xmax = 0
        self.ymax = 0
        self.xmin = 1e9


    # ko can quan tam
    def new_block_bb(self):  # ??
        self.xmin = 1e9
        self.xmax = 0
        self.ymax = self.cursor[1]
        self.ymin = self.cursor[1]


    def random_next_line(self, num=None):
        # random xuong dong bth hay la xuong dong rong
        if num is None:
            num=1
        else:
            num=num
        if np.random.rand() > 0.5:
            self.next_line(num)
        else:
            self.next_wide_line(num)


    # ko can quan tam
    def get_block_bbox(self, label):
        # self.ymax = self.cursor[1]
        self.blocks.append((self.xmin, self.ymin, self.xmax, self.ymax, label))
        bb = (self.xmin, self.ymin, self.xmax, self.ymax)
        # print(bb)
        # self.draw.rectangle(bb , outline = "green")
        # self.draw.text(bb[:2] , label , fill = "black")

    # luu toa do cac box ra file json
    def to_json(self, labels=None):
        h, w = self.img_height, self.img_width
        json_dicts = {'shapes': [], 'imagePath': self.img_name,
                      'imageData': None, 'imageHeight': h, 'imageWidth': w}

        for field in self.fields:
            tl = [field["xmin"], field["ymin"]]
            bl = [field["xmin"], field["ymax"]]
            br = [field["xmax"], field["ymax"]]
            tr = [field["xmax"], field["ymin"]]

            if tl[0] > self.img_width or tl[1] > self.img_height:
                continue

            coords = [tl, bl, br, tr]
            # if field["type"] != "outlier" and "marker" not in field["type"]:
            #     type = "text"
            # else:type = field["type"]
            type = field["type"]

            json_dicts["shapes"].append(
                {'label': type, "text": field["text"], 'points': coords, 'shape_type': 'polygon', 'flags': {}})
  
        json_path = self.dst+self.img_name[:-4] + ".json"
        # print(json_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_dicts, f)



    def save(self):
        self.merge_block()  # gen toa do cac block
        path = self.dst + self.img_name
        self.image = self.image.convert('RGB')
        self.image.save(path) # luu anh
        self.to_json() # luu json
        self.to_xml() # luu xml


    def to_xml(self):
        w, h = self.img_width, self.img_height
        imgname = self.img_name
        xml_path = self.dst + self.img_name[:-4] + ".xml"
        boxes = []
        labels = []
        OCR = []
        # labels_list = []
        for field in self.fields:

            coords = [field["xmin"], field["ymin"],
                      field["xmax"], field["ymax"]]

            if coords[0] > self.img_width or coords[1] > self.img_height:
                continue

            type = field["type"]
            if type not in labels_list:
                labels_list.append(type)
            ocr = field["text"]
            # coords = self.widen_box(coords[0], coords[1], coords[2],  coords[3])

            boxes.append(coords)
            labels.append(type)
            OCR.append(ocr)

        root = ET.Element('annotations')
        filename = ET.SubElement(root, 'filename')
        filename.text = imgname
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        width.text, height.text = str(w), str(h)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'
        for box, label, ocr in zip(boxes, labels, OCR):

            obj = ET.SubElement(root, 'object')
            name = ET.SubElement(obj, 'name')
            name.text = label
            bndbox = ET.SubElement(obj, 'bndbox')
            obj_ocr = ET.SubElement(obj, "OCR")
            obj_ocr.text = u"{}".format(ocr)

            xmin, ymin = ET.SubElement(bndbox, 'xmin'), ET.SubElement(bndbox, 'ymin')
            xmax, ymax = ET.SubElement(bndbox, 'xmax'), ET.SubElement(bndbox, 'ymax')
            xmin.text, ymin.text, xmax.text, ymax.text = [str(b) for b in box]


        ET.ElementTree(root).write(xml_path, encoding="UTF-8")


    def next_line(self, num=1):
        self.cursor[0] = self.left_margin + np.random.randint(0, 20)

        for i in range(num):
            self.cursor[1] += self.line_height
            self.cursor[1] += np.random.randint(-5, 5)
            self.line += 1
    

    def next_small_line(self, num=1):
        self.cursor[0] = self.left_margin + np.random.randint(0, 20)

        for i in range(num):
            self.cursor[1] += self.line_height - 20
            self.cursor[1] += np.random.randint(-5, 5)
            self.line += 1
            

    def next_wide_line(self, i=1):
        self.cursor[0] = self.left_margin + np.random.randint(0, 20)

        for i in range(i):
            self.cursor[1] += self.line_height + 10
            self.cursor[1] += np.random.randint(-5, 5)
            self.line += 1


    def write(self, text, char_font="normal", ink=None, bold=False, size=None, cursor=None):
        # ink = (rgb)

        if ink is None:
            if not bold:
                ink = randink()
            else:
                ink = randink(bold=True)
        if size == None:
            size = self.font_scale
        if cursor == None:
            cursor = self.cursor
            self.cursor = list(self.cursor)

        font = char_font

        cursor = list(cursor)
        # check if self.viet_tay exist

        while self.cursor[0] + self.get_text_length(text, font) > self.img_width:
            text = text[:-1]
        self.draw.text(cursor, text, font=font, fill=ink)

        return text
        

        


    def merge_bb(self, field_name="date_1", cursor=None, block_type=''):
        xmin = 1e9
        ymin = 1e9
        xmax = 0
        ymax = 0
        text = ''
        ls_remove = []
        for field_idx, field in enumerate(self.fields):
            type = field["type"]
            if type != field_name:
                continue
            coords = [field["xmin"], field["ymin"],
                      field["xmax"], field["ymax"]]
            
            # cursor để cho biết vị trí dòng, sẽ ko merge những bb nằm trên dòng khác nhau 
            if cursor is not None:
                field_xmin, field_ymin, field_xmax, field_ymax = coords
                if not cursor[1] - 20 < field_ymin < cursor[1] + 20:
                    continue

            xmin = min(xmin, coords[0])
            xmax = max(xmax, coords[2])
            ymin = min(ymin, coords[1])
            ymax = max(ymax, coords[3])
            text += field["text"]
            ls_remove.append(field_idx)
            
        # remove
        for idx in sorted(ls_remove, reverse=True):
            del self.fields[idx]

        # xmin, ymin, xmax,ymax la toa do cuoi cung cua bb đã gộp
        _field = {}
        _field["xmin"] = xmin
        _field["ymin"] = ymin
        _field["xmax"] = xmax
        _field["ymax"] = ymax
        _field["type"] = field_name
        _field["text"] = u"{}".format(text)
        _field["block_type"] = block_type

        self.fields.append(_field)

    
    def merge_block(self):
        
        
        for block_type in self.block_types:
            xmin = 1e9
            ymin = 1e9
            xmax = -1e9
            ymax = -1e9

            for field_idx, field in enumerate(self.fields):
                if 'block_type' in field.keys():
                    if field["block_type"] == block_type:
                        xmin = min(xmin, field["xmin"])
                        xmax = max(xmax, field["xmax"])
                        ymin = min(ymin, field["ymin"])
                        ymax = max(ymax, field["ymax"])

            if xmin != 1e9 and ymin != 1e9 and xmax != -1e9 and ymax != -1e9:
                # if block_type == 'block_sign':
                #     xmin = min(xmin, self.stamp_area[0])
                #     xmax = max(xmax, self.stamp_area[2])
                #     ymin = min(ymin, self.stamp_area[1])
                #     ymax = max(ymax, self.stamp_area[3])

                #     xmin = min(xmin, self.sign_area[0])
                #     xmax = max(xmax, self.sign_area[2])
                #     ymin = min(ymin, self.sign_area[1])
                #     ymax = max(ymax, self.sign_area[3])
                
                if block_type == 'block_info':
                    xmin, ymin, xmax, ymax = self.widen_block(xmin, ymin, xmax, ymax, block_info=True)
                else:
                    xmin, ymin, xmax, ymax = self.widen_block(xmin, ymin, xmax, ymax)

                _field = {}
                _field["xmin"] = xmin
                _field["ymin"] = ymin
                _field["xmax"] = xmax
                _field["ymax"] = ymax
                _field["type"] = block_type
                _field["text"] = ''

                self.fields.append(_field)
                


    def get_field_coord(self, text, fields, fields_list=[], font=None, poi=False, cursor=None, block_type='', cut=True):
        """
            text = "Ba Le thi mai linh"
            fields = ["ba", "le thi mai linh"]
            fields_list = ["gender", "transfer_name"]

            text = 'ngày 19 tháng 12 năm 2020'
            fields = ['19', '12', '2020']
            field_list = ['field1', 'field2', 'field3']
        """

        "only for 1 line text"

        if cursor == None:
            cursor = self.cursor
        if font == None:
            font = self.normal
        outlier = text

        idx2search = 0 # idx to start to search for field
        for i, field in enumerate(fields):
            field = str(field)
            words = field.split(" ")
            # if poi:
            #     print(words)
            
            # A-B gộp lại làm 1 box
            if '-' in words:
                # print(words)
                indices = [i for i, x in enumerate(words) if x == "-"]
                for idx in sorted(indices, reverse=True):
                    # concat text before and after '-'
                    words[idx-1] = words[idx-1] + ' ' + '-' + ' ' + words[idx+1]
                    # remove
                    del words[idx+1]
                    del words[idx]

            if field not in text:
                continue
            else:
                start = text.index(field, idx2search) # idx dau tien cuar field trong text

                # prevent more than one occurences of a field in 1 line
                # import re
                # field_idx_ls = [m.start() for m in re.finditer(field, text)]       
                # if len(field_idx_ls) == 1:
                #     start = field_idx_ls[0]
                # else:
                #     return -1

            # print(text.split(" "))

            end = start + len(field)  # idx cuoi cung cuar field trong text
            idx2search = start + len(field)
            # field = "ba" => outlier = "__ le thi mai linh"
            outlier = outlier[:start] + " " * len(field) + outlier[end:]

            offset = list(cursor).copy()
            # đưa top-left của text, nội dung và font của text vào => suy ra được bb của text đó
            bb_start = self.draw.textbbox(offset, text[:start], font)  # bb cua phan truoc field

            # bb_end = bb_start + self.get_text_length(field)
            text_bbox = self.draw.textbbox(
                (bb_start[2], cursor[1]), field, font)  # bb cuar field

            bb = text_bbox
            idx = 0  # char idx in words
            for word in words:
                n = len(word)
                if n == 0:
                    idx += 1
                    continue

                word_index = field.index(word, idx)
                if idx != word_index:
                    pass

                word_bb_start = self.draw.textbbox(
                    (bb[0], bb[1]), field[:word_index], font)  # bb cua phan truoc word trong field

                
                word_bb = self.draw.textbbox(
                    (word_bb_start[2], cursor[1]), word, font)  # bb cua word trong field
                
                if True:
                    # prevent loi ra le phai
                    flag_loi_ra_le_phai = False
                    while word_bb[2] - self.get_last_length(word, font) // 4 > self.img_width:
                        flag_loi_ra_le_phai = True
                        print('word hit deleted: ', word)
                        word = word[:-1]

                        word_bb = self.draw.textbbox(
                            (word_bb_start[2], cursor[1]), word, font)  # bb cua word trong field
                    
                    
                    # prevent lot xuong duoi
                    if word_bb[3] - self.get_text_height(word, font) // 5 > self.img_height:
                        continue

                self.xmax = max(self.xmax, word_bb[2])
                self.xmin = min(self.xmin, word_bb[0])
                self.ymax = max(self.ymax, word_bb[3]) 

                # widen box
                xmin, ymin, xmax, ymax = self.widen_box(word_bb[0], word_bb[1], word_bb[2], word_bb[3], cut=cut)
                _field = {}
                _field["xmin"] = xmin
                _field["ymin"] = ymin
                _field["xmax"] = xmax
                _field["ymax"] = ymax
                _field["type"] = fields_list[i]
                if fields_list[i] == 'chudoc':
                    print('chudoc here')
                _field["text"] = u"{}".format(word)
                _field["block_type"] = block_type
                # print(_field["text"])            
                # print('field type: ',fields_list[i])
                self.fields.append(_field)

                idx += len(word) + 1

                if flag_loi_ra_le_phai:
                    print('final word: ', word)
                    break

        # print('outlier: ', outlier)
        self.get_outlier_coord(outlier, text, font, cursor=cursor, block_type=block_type)

        return 0


    def get_outlier_coord(self, outlier, text, font=None, cursor=None, block_type=''):
        # sau khi get hết các field trong 1 line thì các chữ còn lại là outlier => get outlier
        if cursor == None:
            cursor = self.cursor

        if font is None:
            font = self.normal
        # if font == "Bold":
        lines = outlier.split("\n")
        text_lines = text.split("\n")

        line_tl = list(cursor).copy()
        for i, line in enumerate(lines):
            words = line.split(" ")
            text_line = text_lines[i]

            # print(words)
            idx = 0
            for word in words:
                n = len(word)
                if n == 0 or "-" in word:
                    idx += n
                    continue

                # print(text_line )
                # print(word)

                start = text_line.index(word, idx)

                end = start + n
                text_bb = self.draw.textbbox(
                    cursor, text_line[:start], font)
                # bb_end =  self.get_text_length(text_line[:end] , font)
                # if font == self.Big_Bold:
                #     bb = [line_tl[0] + bb_start - 6 , line_tl[1] - 3 , line_tl[0]  + bb_end,line_tl[1] + line_width + 65]
                # else:
                #     bb = [line_tl[0] + bb_start - 6 , line_tl[1] - 3 , line_tl[0]  + bb_end,line_tl[1] + line_width + 20]

                bb = self.draw.textbbox(
                    (text_bb[2], cursor[1]), word, font)
                # self.draw.rectangle(bb , outline = "blue")
                self.xmax = max(self.xmax, bb[2])
                self.xmin = min(self.xmin, bb[0])


                xmin, ymin, xmax, ymax = self.widen_box(bb[0], bb[1], bb[2], bb[3])
                _field = {}
                _field["xmin"] = xmin
                _field["ymin"] = ymin
                _field["xmax"] = xmax
                _field["ymax"] = ymax
                _field["type"] = 'outlier'
                _field["text"] = u"{}".format(word)
                _field["block_type"] = block_type
                # print(_field["text"])
                self.fields.append(_field)
                idx += n + 1
            line_tl[0] = 120
            line_tl[1] += 58


    def widen_box(self, xmin, ymin, xmax, ymax, factor=1.1, cut=True):     
        center = [(xmax + xmin) / 2, (ymax + ymin) / 2]
        # print("original" , xmin , ymin , xmax , ymax)
        # print(center)
        xmin = int(1.05 * ( xmin - center[0] ) + center[0])
        xmax = int( 1.03 * ( xmax- center[0] ) + center[0])
        ymin = int(factor * (ymin - center[1]) + center[1])
        ymax = int(factor * (ymax - center[1]) + center[1])

        if cut:
            if xmax > self.img_width:
                xmax = self.img_width
            if ymax > self.img_height:
                ymax = self.img_height

        # print(xmin , ymin , xmax , ymax)
        return xmin, ymin, xmax, ymax

    
    def widen_block(self, xmin, ymin, xmax, ymax, factor=1.05, block_info=False):
        center = [(xmax + xmin) / 2, (ymax + ymin) / 2]
        # print("original" , xmin , ymin , xmax , ymax)
        # print(center)
        if block_info:
            print('block info here')
            xmin = xmin - 8
        else:
            xmin = int(factor * ( xmin - center[0] ) + center[0])
            
        xmax = int( factor * ( xmax- center[0] ) + center[0])
        ymin = int(factor * (ymin - center[1]) + center[1])
        ymax = int(factor * (ymax - center[1]) + center[1])

        if xmax > self.img_width:
            xmax = self.img_width
        if ymax > self.img_height:
            ymax = self.img_height

        # print(xmin , ymin , xmax , ymax)
        return xmin, ymin, xmax, ymax


    def get_marker_coord(self, text, fields, fields_list, font=None, cursor=None, block_type=''):
        "For 1 line text"
        if cursor== None:
            cursor = self.cursor
        if font == None:
            font = self.normal

        outlier = text

        for i, field in enumerate(fields):
            field = str(field)
            words = field.split(" ")
            start = text.index(field)

            # print(text.split(" "))

            end = start + len(field)
            outlier = outlier[:start] + " " * len(field) + outlier[end:]
            text_bb = self.draw.textbbox(cursor, text[:start], font)
            bb = self.draw.textbbox((text_bb[2], cursor[1]), field, font)


            # self.draw.rectangle(bb , outline = "red")
            self.xmax = max(self.xmax, bb[2])
            self.xmin = min(self.xmin, bb[0])


            xmin, ymin, xmax, ymax = self.widen_box(bb[0], bb[1], bb[2], bb[3])
            _field = {}
            _field["xmin"] = xmin
            _field["ymin"] = ymin
            _field["xmax"] = xmax
            _field["ymax"] = ymax
            _field["type"] = fields_list[i]
            _field["text"] = u"{}".format(fields[i])
            _field["block_type"] = block_type

            # print(_field["text"])
            self.fields.append(_field)
            #     idx += len(word)


    def get_text_length(self, text, font=None):
        """
            For time_new_roman , upper case is wider than lower case , no italic, bold
        """
        if font == None:
            font = self.normal

        l = 0
        for i in range(len(text)):
            # print(font.getsize(text[i])[1] )
            l += font.getsize(text[i])[0]

            # print(self.font.getsize(text[i])[0] , text[i])
        return l
    

    def get_last_length(self, text, font):
        if font==None:
            font = self.normal
        
        return font.getsize(text[-1])[0]


    def get_text_height(self, text, font=None):
        if font == None:
            font = self.normal
        
        max = 0
        for i in range(len(text)):
            if font.getsize(text[i])[1] > max:
                max = font.getsize(text[i])[1]
        
        return max


    def get_text_bbox(self, text, font=None, cursor=None):
        if cursor== None:
            cursor = self.cursor
        if font == None:
            font = self.normal
        bbox = self.draw.textbbox(self.cursor, text, font)



    def fake_BG(self):
        if np.random.rand() > 0.7:
            bg_path = np.random.choice(glob.glob("bg/*"))
            bg = Image.open(bg_path)
            # start = time.time()
            i = np.random.randint(4)

            bg = bg.resize([self.img_width, self.img_height])

            bg = np.array(bg)  # .getdata()).reshape(bg.size[1], bg.size[0], 3)

            image = np.array(self.image)

            bg = bg * 1./255
            image = (bg * image).astype(np.uint8)

            self.image = Image.fromarray(image)


    def perspective_transform(self):
        # new_BG = np.zeros((self.img_height , self.img_width , 3))
        new_width = np.random.randint(2500, 2600)
        new_height = np.random.randint(3800, 4000)
        # print(new_width , new_height)
        src_image = np.array(self.image)

        tl = [np.random.normal(new_width / 30, new_width / 100),
              np.random.normal(new_height / 30, new_height / 100)]
        tr = [new_width - np.random.normal(new_width / 30, new_width / 200),
              np.random.normal(new_height / 30, new_height / 100)]
        br = [new_width - np.random.normal(new_width / 30, new_width / 200),
              new_height - np.random.normal(new_height / 30, new_height / 100)]
        bl = [np.random.normal(new_width / 30, new_width / 200), new_height -
              np.random.normal(new_height / 30, new_height / 100)]

        # self.draw.text(tl , "0" , font =self.font , fill = "white")
        # self.draw.text(tr , "1" , font = self.font , fill = "white")
        # self.draw.text(br , "2" , font = self.font , fill = "white")
        # self.draw.text(bl , "3" , font = self.font , fill = "white")

        coords = np.array([tl, tr, br, bl])
        coords[coords < 0] = 0

        # print(coords)
        pts2 = np.float32(coords)  # dst coordinate
        width, height = self.image.size
        pts1 = np.float32([[0, 0], [width-1, 0],
                           [width-1, height-1], [0, height-1]])  # src coordinate (chính là tất cả ảnh)
        matrix, _ = cv2.findHomography(pts1, pts2)
        # print(matrix)
        result = cv2.warpPerspective(
            src_image, matrix, (new_width, new_height))
        self.image = Image.fromarray(result)
        # new_width = coords[:,0].max() + max(np.random.normal(new_width // 300 , new_width / 200),0)
        # new_height = coords[:,1].max() + max(np.random.normal(new_height // 300 , new_height /200 ),0)
        self.transform_bbox(matrix)
        # print(matrix @ np.array([self.image.size[0], self.image.size[1], 1]))
        self.img_width = new_width
        self.img_height = new_height
        # print(self.img_width, self.img_height)
        # print(self.image.size)
        # self.image = self.image.crop((0,0,new_width , new_height))



    def crop(self):
        tl = [0, 0]
        rb = [self.img_width, self.cursor[1] + 100]
        bb = tl + rb
        self.image = self.image.crop(bb)


    def break_content(self, content, thr=100):

        lines = []
        pv_line = 0
        for i in range(len(content) // thr):
            try:
                nl = content.index(" ", thr * (i+1))
                line = content[pv_line: nl]
                pv_line = nl
                # print(line)
                lines.append(line)

            except:
                break

        lines.append(content[pv_line:])
        return lines


    def break_content_with_field(self, content, fields, fields_type, char_per_line=108):
        new_fields = []
        lines = []
        previous_line = 0  
        has_nextline_field = False
        # print(fields, fields_type)
        for i in range(len(content) // char_per_line):
            newline_idx = content.index(" ", char_per_line * (i+1))  
            field_inline = []
            if i==0:
                line = content[previous_line: newline_idx]
            else: 
                line = content[previous_line+1:newline_idx]

            # print(line)
            lines.append(line)
            # print(line)
            # print("has nl field", has_nextline_field)
            if has_nextline_field:
                field_inline.append(field_nextline)
                # print(field_inline)
            # field_type_inline = []
            has_nextline_field = False
            for j, field in enumerate(fields):
                field_type = fields_type[j]
                # print("field",field)

                # if j < len(fields) - 1:
                #     field_idx = content.index(field)
                # else:
                #     # print('vao last field roi day ')
                #     import re
                #     field_idx_ls = [m.start() for m in re.finditer(field, content)]         
                #     field_idx = field_idx_ls[-1]
                #     # print('field_idx_last_field', field_idx)

                field_idx = content.index(field)


                # print(nl, field_idx)
                if field_idx + len(field) <= newline_idx and field_idx >= previous_line+1:
                    field_inline.append((field, field_type))
                    # print("-------------break_content_with_field______________")
                    # print("field_inline:",(field , field_type , nl , field_idx))
                elif field_idx < newline_idx and field_idx + len(field) > newline_idx:
                    inline_field = field[: newline_idx - field_idx ]
                    field_inline.append((inline_field, field_type))
                    # print("field_in_line",(inline_field, field_type , nl , field_idx))

                    field_nextline = (field[newline_idx - field_idx + 1:], field_type)
                    # print("field_next_line",(field_nextline , field_type ))
                    has_nextline_field = True
                    # print(has_nextline_field)
                
                else:
                    continue

            previous_line = newline_idx
            new_fields.append(field_inline)


        # dòng cuối cùng
        lines.append(content[previous_line+1:])
        field_inline = []
        line = content[previous_line+1:]
        # print("last_line",line)
        if has_nextline_field:
            field_inline.append(field_nextline)
        # field_type_inline = []
        for j, field in enumerate(fields):
            field_type = fields_type[j]
            # print(f'fields: {field}, field_type: {field_type}, line: {line}')
            # print("field",field)

            has_nextline_field = False
            if field in line:
                # print('inline roi day')
                field_idx = line.index(field)
                # if j != len(fields)-1:
                if True:
                    # print('append roi day')
                    field_inline.append((field, field_type))
            # print('field_idx_last_field', field_idx)

            
            
        # pv_line = nl
        new_fields.append(field_inline)

        # print("return lines:",lines)
        # print("return fields:",new_fields)
        
        return lines, new_fields



    def fake_glare(self):
        if np.random.rand() > 0.8:
            i = np.random.randint(2, 4)
            for _ in range(i):
                glare_path = np.random.choice(glob.glob('flare/flare.png'))
                glare_img = Image.open(glare_path).resize((300, 300))
                self.image = self.image.convert("RGBA")
                glare_img = glare_img.convert("RGBA")
                # random position
                position = (np.random.randint(0, self.img_width - 800), np.random.randint(0, self.img_height - 800))
                self.image.paste(glare_img, position, glare_img)
                self.image = self.image.convert("RGB")
                self.draw = ImageDraw.Draw(self.image)

    
    def fake_stamp(self):
        offset_x = int(np.random.randint(-70, 40))
        offset_y = int(np.random.randint(-40, 40))

        self.stamp_area = [700 + offset_x,  1350 + offset_y, 1000 + offset_x ,  1650 + offset_y]

        stamp_path = np.random.choice(glob.glob("stamp/*.png"))

        stamp = cv2.imread(stamp_path)
        stamp = random_rotate(stamp)
        # stamp = constrast_stretching(stamp)

        self.image = self.image.convert('RGB')
        src = np.array(self.image) # current self.image
        stamp = cv2.resize(stamp , (self.stamp_area[2] - self.stamp_area[0] , self.stamp_area[3] - self.stamp_area[1])) # resize stamp
        stamp = cv2.cvtColor(stamp, cv2.COLOR_BGR2RGB) # convert to RGB
        
        # filter white pixel
        mask = np.any((stamp[:,:,1:] < 200),axis= -1)
        mask = np.expand_dims(mask , -1)

        print(mask.shape)
        print(np.unique(mask))
        stamp = mask * list(self.stamp_ink)  * np.random.randint(600, 900) / 1000.
        print(stamp.shape)

        # cho nao mask=1 thi lay cua stamp
        # cho nao mask=0 thi lay cua anh goc
        print('stamp shape', stamp.shape)
        print('mask shape: ', mask.shape)
        stamp = stamp * mask + src[self.stamp_area[1] : self.stamp_area[3] , self.stamp_area[0] : self.stamp_area[2]] * (1-mask) 

        # find pixel position in mask that has value = 1
        temp = np.where(mask == 1)
        pos = [temp[0][0], temp[1][0]]
        self.stamp_ink = stamp[pos[0], pos[1]]
        # convert to int
        self.stamp_sign_ink = tuple([int(x) for x in self.stamp_ink]) 

        # src[sign1_rec[1] : sign1_rec[3] , sign1_rec[0] : sign1_rec[2]] =  stamp
        stamp = Image.fromarray(stamp.astype(np.uint8)).convert('RGBA')
        # blend stamp with src
        temp = src[self.stamp_area[1] : self.stamp_area[3] , self.stamp_area[0] : self.stamp_area[2]]
        temp = Image.fromarray(temp).convert('RGBA')
        factor = rand_normal(0.7, 0.1)
        temp = Image.blend(temp, stamp, factor)
        temp = temp.convert('RGB')
        src[self.stamp_area[1] : self.stamp_area[3] , self.stamp_area[0] : self.stamp_area[2]] = np.array(temp)

        

        self.image = Image.fromarray(src)
        self.draw = ImageDraw.Draw(self.image)


    

        
    def fake_finger(self):

        finger_path = np.random.choice(glob.glob("finger/*.png"))
        finger = cv2.imread(finger_path)
        

        offset_x = int(np.random.normal(-20,20))
        offset_y = int(np.random.normal(-20,20))

        flag = np.random.choice([0,1])
        if flag == 0: # right
            tl0 = np.random.randint(self.img_width-200, self.img_width-100)
            tl1 = np.random.randint(0, self.img_height-300)
        else:
            tl0 = np.random.randint(-30, -20)
            tl1 = np.random.randint(0, self.img_height-300)
            # rotate 180 degree
            finger = cv2.rotate(finger, cv2.ROTATE_180)

        tl = [tl0, tl1]
        br = [tl[0]+300, tl[1]+300]
        sign1_rec = [tl[0] + offset_x,  tl[1] + offset_y, br[0] + offset_x ,  br[1] + offset_y]
        
        finger = random_rotate(finger)
        # signal = constrast_stretching(signal)

        src = np.array(self.image) # current self.image
        # pad src by 300 at each side
        num_pad = 300
        src = np.pad(src, ((num_pad,num_pad),(num_pad,num_pad),(0,0)), 'constant', constant_values=255)
        finger = cv2.resize(finger , (sign1_rec[2] - sign1_rec[0] , sign1_rec[3] - sign1_rec[1])) # resize stamp
        finger = cv2.cvtColor(finger, cv2.COLOR_BGR2RGB) # convert to RGB
        # split finger vertically
        
        # filter white pixel
        
        white = np.all((finger[:,:,:] > 170),axis= -1)
        # revert mask
        mask = np.logical_not(white)
        mask = np.expand_dims(mask , -1)
        # print(np.unique(mask))

        print(mask.shape)
        print(np.unique(mask))
        # finger = mask * list(self.finger_ink) * l
        print(finger.shape)

        # cho nao mask=1 thi lay cua stamp
        # cho nao mask=0 thi lay cua anh goc
        finger = finger * mask + src[sign1_rec[1]+num_pad : sign1_rec[3]+num_pad, sign1_rec[0]+num_pad : sign1_rec[2]+num_pad] * (1-mask)

        src[sign1_rec[1]+num_pad : sign1_rec[3]+num_pad , sign1_rec[0]+num_pad : sign1_rec[2]+num_pad] =  finger

        # restore src
        src = src[num_pad:-num_pad, num_pad:-num_pad]
        self.image = Image.fromarray(src)
        self.draw = ImageDraw.Draw(self.image)



    


    





    def fake_blur(self):
        # blur random area
        if np.random.rand() > 0.6:
            i = np.random.randint(1, 5)
            for _ in range(i):
                tl = [np.random.randint(0, self.img_width-100), np.random.randint(0, self.img_height-100)]
                br = [e + np.random.randint(100, 300) for e in tl]
                area = (tl[0], tl[1], br[0], br[1])
                ic = self.image.crop(area)
                ic = ic.filter(ImageFilter.GaussianBlur(radius=np.random.randint(1, 3)))
                self.image.paste(ic, area)

    
    def ep_plastic(self):
        roi_img = self.image.convert('RGBA')
        roi_w, roi_h = roi_img.size
        pad = min(roi_w, roi_h) // 20
        val = (np.random.randint(150, 230), ) + (np.random.randint(150, 230), ) + (np.random.randint(150, 230), ) + (np.random.randint(254, 255), )
        roi_plastic = Image.new('RGBA', (roi_w+pad*2, roi_h+pad*2), val)
        roi_plastic.paste(roi_img, (pad, pad), roi_img)

        mask = Image.new('L', roi_plastic.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, roi_plastic.size[0], roi_plastic.size[1]), radius=np.random.randint(60, 100), fill=np.random.randint(120, 151))
        draw.rectangle((pad, pad, pad+roi_w, pad+roi_h), fill=255)
        roi_plastic.putalpha(mask)

        bg_path = '/home/fiores/Downloads/BG/val/' + np.random.choice(os.listdir('/home/fiores/Downloads/BG/val'))
        bg = Image.open(bg_path).convert('RGBA').resize(roi_plastic.size)
        bg.paste(roi_plastic, (0, 0), roi_plastic)
        self.image = bg

        for field_idx, field in enumerate(self.fields):
            field['xmin'] += pad
            field['ymin'] += pad
            field['xmax'] += pad
            field['ymax'] += pad

        self.img_height += 2*pad
        self.img_width += 2*pad

    def vien_trang(self):
        img = np.array(self.image)
        pad_value = np.random.randint(200, 255)
        pad_size = min(img.shape[:2]) // 25
        padded_roi = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[pad_value+randint(-20, 20), pad_value+randint(-20, 20), pad_value+randint(-20, 20)])
        self.image = Image.fromarray(padded_roi)

        for field_idx, field in enumerate(self.fields):
            field['xmin'] += pad_size
            field['xmax'] += pad_size
            field['ymin'] += pad_size
            field['ymax'] += pad_size
        
        self.img_height += 2*pad_size
        self.img_width += 2*pad_size


    def fake_general_image(self):
        # # zoom image
        # if np.random.rand() > 0.7:
        #     num_crop = np.random.randint(30, 70)
        #     self.image = self.image.crop((num_crop, num_crop, self.img_width-num_crop, self.img_height-num_crop))
        #     self.image = self.image.resize((self.img_width, self.img_height))

        # blur
        if np.random.rand() > 0.7:
            self.image = self.image.filter(ImageFilter.GaussianBlur(radius=np.random.randint(10, 15)/10.))
        
        # noise
        if np.random.rand() > 0:
            from skimage.util import random_noise
            image = np.array(self.image)
            var = (np.random.normal(0.3, 0.1) /10.) **2
            noise_img = random_noise(image, mode='gaussian', var=var)
            noise_img = (255*noise_img).astype(np.uint8)

            self.image = Image.fromarray(noise_img)
        
        
        # contrast
        if np.random.rand() > 0.7:
            self.image = ImageEnhance.Contrast(self.image).enhance(np.random.normal(1, 0.25))
        
        # brightness
        if np.random.rand() > 0.7:
            self.image = ImageEnhance.Brightness(self.image).enhance(randint(85, 115)/100.)


# if __name__ == '__main__':
#     i=0
#     while i<100:
#         print("_______________________________________________________________________________________________________________")
#         try:
#             faker = OfficialID()
#             print(faker.characters)
#             faker.fake()
#             print(faker.img_name)
#             print("faking")
#             i += 1
#         except Exception as e:
#             # raise e
            
#             print(e)
#             continue
#             # break
