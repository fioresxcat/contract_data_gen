from common import *
import numpy as np
from time import time
import json
import os
import cv2
import PIL.Image as Image
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageFilter as ImageFilter
from PIL import ImageEnhance
import pdb


class SubModule:
    def __init__(self, shape = [1000, 1000], marker_prob = 0.5, down_prob=0.2,
                 marker_font:ImageFont.truetype = None, content_font:ImageFont.truetype = None,
                 markers = [], content = None, label = None, ink = None, augment_prob = 0.5,
                 random_capitalize_prob={
                        'marker': [0.7, 0.25, 0.05],
                        'content': [0.6, 0.3, 0.1]
                 }, module_order_type = 'up|down'):
        self.canvas =  np.full(shape + (3,), 255, dtype=np.uint8)
        self.canvas = Image.fromarray(self.canvas)
        self.shape = shape
        self.fields = []
        self.markers = markers
        self.content = content
        self.has_marker = np.random.rand() < marker_prob
        self.down_prob = down_prob
        self.in_module_position = (0, 0)
        self.marker_font = marker_font
        self.content_font = content_font
        self.cursor = [10, 10]
        self.label = label
        self.draw = ImageDraw.Draw(self.canvas)
        self.ink = ink
        self.default_font_size = 20
        self.augment_prob = augment_prob
        self.random_capitalize_prob = random_capitalize_prob
        self.module_order_type = module_order_type

    def get_shape(self):
        return self.canvas.shape[:2]
        
    def __call__(self):
        # self.canvas =  np.full(self.shape + (3,), 255, dtype=np.uint8)
        # self.canvas = Image.fromarray(self.canvas)

        text = ""
        flag = np.random.choice([1, 2, 3])

        if self.has_marker:
            marker_text = np.random.choice(self.markers)
            marker_text = random_space(marker_text)
            marker_text = random_capitalize(marker_text, self.random_capitalize_prob['marker'])
            if flag == 1:
                marker_text += ': '

            text += marker_text
            if flag == 2:
                divider_text = np.random.randint(1, 5) * ' ' + ':' + np.random.randint(1, 5) * ' '
                text += divider_text

            self.write(font = self.marker_font, text=text)
            self.get_field_coord(text, [marker_text], ['marker_' + self.label], self.marker_font)

        # actual content
        content_text = np.random.choice(self.content)
        content_text = random_capitalize(content_text, self.random_capitalize_prob['content'])
        content_text = random_space(content_text)
        if flag == 3 and text != '':
            content_text = ' :' + content_text

        is_down = False
        if self.module_order_type == 'left|right' and get_num_char(content_text) > 30 and self.label != 'bank_name':
            is_down = True
        elif np.random.rand() > self.down_prob or len(content_text.split()) < 6 or get_num_char(content_text) < 25:
            is_down = False
        else:
            is_down = True
        if not is_down:   # khong xuong dong
            self.cursor[0] += self.marker_font.getsize(text)[0]
            self.write(content_text, self.content_font, bold=np.random.choice([False, True]))
            self.get_field_coord(content_text, [content_text], [self.label], self.content_font)
        else:
            ls_parts = split_text(content_text, factor_range=[0.5, 1]) if self.module_order_type == 'up|down' else split_text(content_text, factor_range=[0.5, 0.7])
            if len(ls_parts) == 2:
                part1, part2 = ls_parts
            else:
                part1, part2 = ls_parts[0], ''
            ## part 1
            self.cursor[0] += self.marker_font.getsize(text)[0]
            self.cursor[1] += np.random.randint(-5, 2)
            self.write(text=part1, font=self.content_font)
            self.get_field_coord(part1, [part1], [self.label], self.content_font)

            ## part 2
            self.cursor[0] = self.cursor[0] * np.random.uniform(0.8, 1.2)
            self.cursor[1] += self.marker_font.getsize(part1)[1] + np.random.randint(0, 5)
            self.write(text=part2, font=self.content_font)
            self.get_field_coord(part2, [part2], [self.label], self.content_font)
        
        self.cut_canvas_to_roi()
        if np.random.rand() < self.augment_prob:
            self.augment()
        self.canvas = np.asarray(self.canvas)

        return self
    
    def augment(self):
        if np.random.random() < 0.5:
            self.canvas = PIL_augment(self.canvas)
        if np.random.random() < 0:
            self.canvas = erode_dilate(self.canvas)
        if np.random.random() < 0.5:
            self.canvas = random_drop_black_pixel(self.canvas)    
        
    def resize(self, new_shape):
        boxes = [text['box'] for text in self.fields]
        self.canvas, boxes = resize(new_shape, self.canvas, boxes)
        for i in range(len(boxes)):
            self.fields[i]['box'] = boxes[i]
        self.shape = self.canvas.shape[:2]
    
    def get_part(self, x, y):
        '''
            Get a part of canvas within (0, x) and (0, y) 
        '''
        canvas = self.canvas[:y, :x]
        texts = []
        boxes = [text['box'] for text in self.fields]
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x2 < x and y2 < y: # Remove box outside of the ROI
                texts.append(self.fields[i])
        part_submodule = SubModule(canvas=canvas)
        part_submodule.texts = texts
        return part_submodule
    

    def cut_canvas_to_roi(self):
        boxes = np.array([text['box'] for text in self.fields])
        xmin = np.min(boxes[:, 0])
        ymin = np.min(boxes[:, 1])
        xmax = np.max(boxes[:, 2])
        ymax = np.max(boxes[:, 3])
        
        self.canvas = self.canvas.crop((xmin, ymin, xmax, ymax))
        
        # recalibrate the coordinates
        for i, field in enumerate(self.fields):
            self.fields[i]['box'][0] -= int(xmin)
            self.fields[i]['box'][1] -= int(ymin)
            self.fields[i]['box'][2] -= int(xmin)
            self.fields[i]['box'][3] -= int(ymin)

            
    def write(self, text: str = None, font = None, bold = None):
            """_summary_

            Args:
                text (list): _description_
                char_font (str, optional): _description_. Defaults to "normal".
                ink (_type_, optional): _description_. Defaults to None.
                bold (bool, optional): _description_. Defaults to False.
                font_size (_type_, optional): _description_. Defaults to None.
                cursor (list, optional): _description_. Defaults to None.
                canvas (np.array, optional): _description_. Defaults to None.

            Raises:
                Exception: _description_
                Exception: _description_

            Returns:
                _type_: _description_
            """

            if self.canvas is None:
                raise Exception("canvas cannot be None")

            if self.ink is None:
                self.ink = randink(bold=bold)

            # pdb.set_trace()
            # print(self.cursor)
            # print(self.get_text_length(text = text, font = font))
            # print(self.canvas.size)
            while self.cursor[0] + self.get_text_length(text = text, font = font)[0] > self.canvas.size[0]:
                text = text[:-1]

            self.draw.text(self.cursor, text=text, font = font, fill=self.ink)
    
    def get_text_length(self, text, font) -> int:
        """_summary_

        Args:
            text (_type_): _description_
            font (ImageFont.truetype, optional): _description_. Defaults to None.

        Raises:
            Exception: _description_

        Returns:
            int: _description_
        """
        if font == None:
            raise Exception("font cannot be None")

        return font.getsize(text)

    def get_field_coord(self, text, fields, fields_list=[], font=None, poi=False, cursor=None, cut=True):
        if cursor == None:
            cursor = self.cursor
        if font == None:
            font = self.normal
        outlier = text

        idx2search = 0 # idx to start to search for field
        for i, field in enumerate(fields):
            field = str(field)
            words = field.split(" ")

            # # A-B gộp lại làm 1 box
            # if '-' in words:
            #     indices = [i for i, x in enumerate(words) if x == "-"]
            #     for idx in sorted(indices, reverse=True):
            #         # concat text before and after '-'
            #         words[idx-1] = words[idx-1] + ' ' + '-' + ' ' + words[idx+1]
            #         # remove
            #         del words[idx+1]
            #         del words[idx]

            if field not in text:
                continue
            else:
                start = text.index(field, idx2search) # idx dau tien cuar field trong text

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
                
                # prevent loi ra le phai
                flag_loi_ra_le_phai = False
                while word_bb[2] - get_last_length(word, font) // 4 > self.canvas.size[0]:
                    flag_loi_ra_le_phai = True
                    # print('word hit deleted: ', word)
                    word = word[:-1]

                    word_bb = self.draw.textbbox(
                        (word_bb_start[2], cursor[1]), word, font)  # bb cua word trong field
                
                
                # prevent lot xuong duoi
                if word_bb[3] - get_text_height(word, font) // 5 > self.canvas.size[1]:
                    continue

                # widen box
                xmin, ymin, xmax, ymax = widen_box(word_bb[0], word_bb[1], word_bb[2], word_bb[3], cut=cut, size=self.canvas.size)
                _field = {}
                _field["box"] = [xmin, ymin, xmax, ymax]
                _field["type"] = fields_list[i]
                _field["text"] = u"{}".format(word)
                self.fields.append(_field)

                idx += len(word) + 1

                if flag_loi_ra_le_phai:
                    # print('final word: ', word)
                    break

        # print('outlier: ', outlier)
        self.get_outlier_coord(outlier, text, font, cursor=cursor)

        return 0
    
    def get_outlier_coord(self, outlier, text, font=None, cursor=None):
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

            idx = 0
            for word in words:
                n = len(word)
                if n == 0 or "-" in word:
                    idx += n
                    continue

                start = text_line.index(word, idx)

                end = start + n
                text_bb = self.draw.textbbox(
                    cursor, text_line[:start], font)

                bb = self.draw.textbbox(
                    (text_bb[2], cursor[1]), word, font)

                xmin, ymin, xmax, ymax = widen_box(bb[0], bb[1], bb[2], bb[3], size=self.canvas.size)
                _field = {}
                _field["box"] = [xmin, ymin, xmax, ymax]
                _field["type"] = 'text'
                _field["text"] = u"{}".format(word)
                self.fields.append(_field)
                idx += n + 1
            line_tl[0] = 120
            line_tl[1] += 58



