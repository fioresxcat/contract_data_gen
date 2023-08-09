import sys
import os
sys.path.append(os.getcwd())

from SubModule import SubModule
from common import *
from contract.sub_modules.content import content as all_content
from contract.sub_modules.font import *

SHAPE = (1000, 1000)

######################
### Phần của Chiến 
######################
address_markers = [
    'Add',
    'Address',
    'A/D',
    'A/d',
    'ADD',
    'ADDRESS',
    'Office'
]

bank_address_markers = [
    'Add',
    'Address',
    'A/D',
    'A/d',
    'ADD',
    'ADDRESS',
    'Bank Address',
    'Bank Add',
    'Bank A/d',
]



comname_markers = {
    '1': [
        'THE SELLER',
        'THE BUYER',
        'PARTY A',
        'PARTY B',
        'SELLER',
        'BUYER',
    ],

    '2': [
        'BETWEEN',
        'Company Name',
        'AND',
        'PARTY A (SELLER)',
        'PARTY B (BUYER)',
        'FROM',
        'TO',
    ],

    '3': [
        'MANUFACTURER',
        'EXPORTER',
        'IMPORTER',
        'THE SHIPPER',
    ]
}


class CompanyName(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=0.7, down_prob=0.2, ink=None):
        marker_type = np.random.choice(['1', '2', '3'], p=[0.5, 0.3, 0.2])
        markers = comname_markers[marker_type]
        comname_content = all_content['en_com_name_abrre'] if np.random.rand() < 0.6 else all_content['vn_com_name']
        content = [el for el in comname_content if len(el.split()) >=2] if np.random.rand() < 0.9 else [el for el in comname_content if len(el.split()) <= 2]
        content = [el for el in content if 'bank' not in el.lower()]  # company name ko duoc co bank
        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'company_name', ink)


class Company_Address(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=0.7, down_prob=0.2,  ink=None):
        markers = address_markers
        address_content = all_content['en_com_add'] if np.random.rand() < 0.6 else all_content['vn_unsign_add']
        content = address_content if np.random.rand() <  0.6 else [' ' * randint(3, 10) + el for el in address_content]  # truong address thuong co mot khoang trong rat lon o dau
        content = [el for el in content if len(el.split()) >=2]
        if np.random.rand() < 0.3:
            rand_num = random_number(randint(3, 7))
            content = [el + ' ' + rand_num for el in content]
        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'company_address', ink)


class Bank_Address(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=0.7, down_prob=0.2,  ink=None):
        markers = bank_address_markers
        address_content = all_content['en_com_add'] if np.random.rand() < 0.6 else all_content['vn_unsign_add']
        content = address_content
        content = [el for el in content if len(el.split()) >=2]
        if np.random.rand() < 0.3:
            rand_num = random_number(randint(3, 7))
            content = [el + ' ' + rand_num for el in content]
        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'bank_address', ink)

######################
### Phần của Khôi
######################

class Phone(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=0.5, down_prob = 0, ink=None):
        # content = '' if np.random.random() < 0.5 else '+'
        # for i in range(14):
        #     prob = np.random.random()
        #     if prob < 0.1: content += ' '
        #     elif 0.1 <= prob < 0.2: content += '-'
        #     elif 0.2 <= prob < 0.3: content += '('
        #     elif 0.3 <= prob < 0.4: content += ')'
        #     else:
        #         content += str(randint(0, 10))
        # content = ["".join(content)]

        content = random_number(randint(7, 11))
        if np.random.rand() < 0.5:
            content = '+' + content
        if np.random.rand() < 0.8:
            idx = np.random.randint(0, len(content)-2)
            content = content[:idx] + '(' + content[idx:]
            idx = np.random.randint(idx+2, len(content)-1)
            content = content[:idx] + ')' + content[idx:]
        if np.random.rand() < 0.8:
            # randomly insert '.' or '-' at 2-3 random positions
            char2insert = np.random.choice(['.', '-', ' '])
            pos2insert = np.random.choice(list(range(1, len(content)-1)), np.random.choice([2, 3]), replace=False)
            for pos in pos2insert:
                content = content[:pos] + char2insert + content[pos:]
        content = [content]

        markers = ['Tel', 'Tel No', 'Tel Number', 'Telephone', 'Phone/ Điện thoại', 'Phone', 'City Tel', 'Mobile', 'Mob', 'HP']
        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'phone', ink)

class Fax(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=0.5, down_prob = 0, ink=None):
        # content = '' if np.random.random() < 0.5 else '+'
        # for i in range(14):an
        #     prob = np.random.random()
        #     if prob < 0.1: content += ' '
        #     elif 0.1 <= prob < 0.2: content += '-'
        #     elif 0.2 <= prob < 0.3: content += '('
        #     elif 0.3 <= prob < 0.4: content += ')'
        #     else:
        #         content += str(randint(0, 10))
        # content = ["".join(content)]

        content = random_number(randint(6, 11))
        if np.random.rand() < 0.5:
            content = '+' + content
        if np.random.rand() < 0.7:
            idx = np.random.randint(0, len(content)-2)
            content = content[:idx] + '(' + content[idx:]
            idx = np.random.randint(idx+2, len(content)-1)
            content = content[:idx] + ')' + content[idx:]
        if np.random.rand() < 0.7:
            # randomly insert '.' or '-' at 2-3 random positions
            char2insert = np.random.choice(['.', '-', ' '])
            pos2insert = np.random.choice(list(range(1, len(content)-1)), np.random.choice([2, 3]), replace=False)
            for pos in pos2insert:
                content = content[:pos] + char2insert + content[pos:]
        content = [content]

        markers = ['Fax', 'Fax No', 'Fax No.']
        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font,
         markers, content, 'fax', ink)

class Tax(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=0.5, down_prob = 0, ink=None):    
        num_digit = randint(7, 14)
        content = [random_number(num_digit)]
        markers = ['Tax', 'Tax code', 'Tax No', 'Tax No.', 'MST', 'Company Tax']
        
        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'tax', ink)


class AccountNumber(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=0.5, down_prob = 0, ink=None):
        
        markers=["Account", "A/C No", 'A/c #', 'A/c No VND', 'A/c No USD', 'ACCOUNT NO', 'ACCOUNT NUMBER', 'USD A/C No', 'VND A/C No', 'A/C', 'Bank A/c NO'] \
              + ['IBAN', 'IBAN No']
       
        content = random_number(randint(6, 10))
        if np.random.rand() < 0.45:
            # randomly insert ' ' or '-' at 2-3 random positions
            char2insert = np.random.choice(['-', ' '])
            pos2insert = np.random.choice(list(range(1, len(content)-1)), np.random.choice([2, 3]), replace=False)
            for pos in pos2insert:
                content = content[:pos] + char2insert + content[pos:]
        if np.random.rand() < 0.2:
            # random insert (USD) or (VND) at random position
            char2insert = np.random.choice(['(USD)', '(VND)'])
            pos2insert = np.random.choice(list(range(1, len(content))))
            content = content[:pos2insert] + char2insert + content[pos2insert:]

        content = [content]

        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'account_number', ink)
    

# class RepresentedBy(SubModule):

#     def __init__(self, marker_font, content_font, marker_prob=0.5, down_prob=0.0, ink=None):
        
#         markers=["Represented by", 
#                  'Represented', 
#                  'Presented', 
#                  'Presented By',
#                  "Represented Position",
#                  "Representative",
#                  "Representator",
#                  'Representative by',
#                  'Attn',
#                  'Contact',
#                  'Contact Person',
#                  'Attention']

#         title = np.random.choice(["Mr.", "Mrs.", "Miss"])

#         name = np.random.choice(all_content["en_per_name"])
#         name = name.capitalize() if np.random.random() < 0.8 else name.upper()

#         pos_len = np.random.choice([1, 2, 3], p=[0.5, 0.25, 0.25])
#         pos = np.random.choice(all_content["pos"][pos_len])
        
#         seperator = np.random.choice([' ', '']) * np.random.randint(0, 3) + np.random.choice(['-', ',']) + np.random.choice([' ', '']) * np.random.randint(0, 3)
#         # dash = double_case_prob(0.2) * "-"
#         # _pos = double_case_prob(0.4) * "Position"
#         # __pos = double_case_prob(0.2) * ":"

#         # content = [f"{title} {name.upper()} {dash} {_pos} {__pos} {pos}"]
#         if np.random.rand() < 0.5:  # neu ko co marker position
#             content = [f"{title} {name}{seperator}{pos}"] if np.random.rand() < 0.5 else [f"{name}{seperator}{pos}"]
#         else:
#             marker_pos = np.random.choice(['Position', 'Chức vụ'], p=[0.9, 0.1])
#             content = f'{title} {name}' + ' ' * np.random.randint(3, 7) + f'{marker_pos}: {pos}' if np.random.rand() < 0.5 else f'{name}' + ' ' * np.random.randint(3, 7) + f'{marker_pos}: {pos}'
#             content = [content]

#         super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'represented_name', ink)


class RepresentedBy(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=0.5, down_prob=0.0, ink=None):
        markers=["Represented by", 
                 'Represented', 
                 'Presented', 
                 'Presented By',
                 "Representative",
                 "Representator",
                 'Representative by',
                 'Attn',
                 'Contact',
                 'Contact Person',
                 'Attention']

        title = np.random.choice(["Mr.", "Mrs.", "Miss"])

        name = np.random.choice(all_content["en_per_name"]) if np.random.rand() < 0.7 else np.random.choice(all_content["vn_per_name"])
        name = name.capitalize() if np.random.random() < 0.8 else name.upper()

        content = [f"{title} {name}"] if np.random.rand() < 0.5 else [f"{name}"]
        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'represented_name', ink)



######################
### Phần của Tùng
######################
from faker import Faker
from faker.providers import bank

faker = Faker()
faker.add_provider(bank)

class SwiftCode(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=1, down_prob=0.2, ink=None):
        markers = [
            'swift code',
            'swift',
            'bank swift code',
            'swift address',
            'swift no',
            'swift bic',
            'swift (bic)',
            'swift code (bic)',
        ]

        random_code = faker.swift() if np.random.rand() < 0.9 else faker.swift(length=11, primary=True)
        content = [random_code]
        
        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'swift_code', ink)


class AccountName(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=0.5, down_prob=0.2, ink=None):
        markers = [
            'account name',
            'beneficiary',
            'beneficiary\'s name',
            'beneficiary\'s account name',
            'beneficiary account',
            'name',
            'company name'
        ]
        # content = all_content["en_com_name_abrre"] if np.random.rand() < 0.6 else all_content['vn_com_name']
        comname_content = all_content["en_com_name_abrre"] if np.random.rand() < 0.6 else all_content['vn_com_name']
        content = [el for el in comname_content if len(el.split()) >=2] if np.random.rand() < 0.9 else [el for el in comname_content if len(el.split()) <=2]
        content = [el for el in content if 'bank' not in el.lower()]  # company name ko duoc co bank
        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'company_name', ink)


class BankName(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=1, down_prob=0, ink=None):
        markers = [
            'bank name',
            'bank\'s name',
            'beneficiary bank name',
            'beneficiary bank',
            'beneficiary\'s bank',
            'at the bank',
            'bank',
            'at'
        ]
        content = None
        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'bank_name', ink)

        self.content = [el for el in all_content['en_bank_name'] if 'bank' in el.lower()] if not self.has_marker or np.random.rand() < 0.7 else all_content['en_bank_name']
        self.content = self.content if self.has_marker or np.random.rand() < 0.3 else ['at ' + el for el in self.content]



class RepresentedPosition(SubModule):
    def __init__(self, marker_font, content_font, marker_prob=0.3, down_prob=0.0, ink=None):
        markers=[
            "Represented Position",
            'Position',
            'Chức vụ',
        ]
        pos_len = np.random.choice([1, 2, 3], p=[0.5, 0.25, 0.25])
        pos = np.random.choice(all_content["pos"][pos_len])
        content = [pos]

        super().__init__(SHAPE, marker_prob, down_prob, marker_font, content_font, markers, content, 'represented_position', ink)