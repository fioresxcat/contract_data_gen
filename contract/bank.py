import sys
import os
sys.path.append(os.getcwd())

from sub_modules.sub_modules import *
from Module import Module
import numpy as np

'''
    - required submodules:  bank_name; company_name for "benificiary"; 
                            bank_address; account_number
    - may or may not existed submodules: fax, swift_code, reprenented_name, represented_position, phone, tax
'''

class Bank(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def update_cursor(cursor, submodule, direction='down'):
        h, w = submodule.get_shape()
        if direction == 'down':
            # cursor[0] = cursor[0] #max(2, cursor[0] + randint(-5, 5))
            cursor[1] = cursor[1] + h + randint(5, 20) #x1, y2
        elif direction == 'reset_down':
            cursor[0] = 10 #randint(5, 15)
            cursor[1] = cursor[1] + h + randint(5, 20) #x1, y2
        else:
            cursor[0] = cursor[0] + w + randint(20, 100)
            # cursor[1] = cursor[1] #max(cursor[1] + randint(-5, 5), 2) #x2, y1
        return cursor


    def __call__(self, bank_name, bank_address, company_address, account_number, account_name, swift_code):
        cursor = [10, 10]
        # type = np.random.choice([1, 2, 3, 4, 5, 6])
        # if type == 1:
            # submodules = [bank_name, bank_address, account_number, account_name, swift_code]
        # elif type == 2:
            # submodules = [bank_name, account_number, account_name, bank_address, swift_code]
        # elif type == 3:
            # submodules = [bank_name, account_name, account_number, bank_address, swift_code]
        # elif type == 4:
            # submodules = [bank_name, bank_address, swift_code, account_number, account_name]
        # elif type == 5:
            # submodules = [bank_name, bank_address, account_name, account_number, swift_code]
        # elif type == 6:
            # submodules = [swift_code, account_name, account_number, bank_address]
            # np.random.shuffle(submodules)
            # submodules = [bank_name] + submodules
        bank_skip_prob_dict = {
            'BankName': 0,
            'Bank_Address': 0.3,
            'AccountNumber': 0,
            'AccountName': 0.5,
            'SwiftCode': 0.3,
            'Company_Address': 0.6,
        }
        flag = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.15]*4 + [0.08]*5)
        if flag == 1:
            submodules = [bank_name, bank_address, account_name, company_address, account_number, swift_code]
        elif flag == 2:
            submodules = [bank_name, bank_address, account_name, company_address, swift_code, account_number]
        elif flag == 3:
            submodules = [bank_name, account_number, swift_code, bank_address, account_name, company_address]
        elif flag == 4:
            submodules = [bank_name, swift_code, account_number, bank_address, account_name, company_address]
        elif flag == 5:
            submodules = [account_name, company_address, bank_name, bank_address, account_number, swift_code]
        elif flag == 6:
            submodules = [account_name, company_address, bank_name, bank_address, swift_code, account_number]
        elif flag == 7:
            submodules = [account_name, company_address, bank_name, swift_code, bank_address, account_number]
        elif flag == 8:
            submodules = [account_name, company_address, account_number, bank_name, swift_code, bank_address]
        elif flag == 9:
            submodules = [account_name, company_address, account_number, bank_name, bank_address, swift_code]

        # skip some submodules
        ls_idx2del = []
        for i, submodule in enumerate(submodules):
            if np.random.rand() < bank_skip_prob_dict[submodule.__class__.__name__]:
                ls_idx2del.append(i)
        submodules = [submodule for i, submodule in enumerate(submodules) if i not in ls_idx2del]

        # chuyenr account number len dau
        if np.random.rand() < 0.25:  
            submodules.remove(account_number)                
            submodules = [account_number] + submodules

        # remove company address if company name is not existed
        if company_address in submodules and account_name not in submodules:
            submodules.remove(company_address)


        for i, submodule in enumerate(submodules):
            self.__paste__(submodule, cursor)
            cursor = self.update_cursor(cursor, submodule, 'down')

        # Draw now
        max_box = np.max([sub['box'] for sub in self.submodules], 0)
        xmax, ymax = max_box[2] + 5, max_box[3] + 5
        self.canvas = np.full((ymax, xmax, 3), 255, dtype=np.uint8)
        for i, sub in enumerate(self.submodules):
            x1, y1, x2, y2 = sub['box']
            self.canvas[y1:y2, x1:x2] = sub['submodule'].canvas

if __name__ == '__main__':
    import cv2
    for i in range(10):
        font = Font(font_scale=20)
        font_normal, font_bold, font_italic = font.get_font('normal'), font.get_font('bold'), font.get_font('italic')
        bank = Bank()
        bank_name = BankName(font_normal, font_normal,marker_prob=0.7, down_prob=0)()
        bank_address = Bank_Address(font_normal, font_normal,marker_prob=0.7, down_prob=0.2)()
        account_number = AccountNumber(font_normal, font_normal, marker_prob=1)()
        account_name = AccountName(font_normal, font_normal, marker_prob=0.7, down_prob=0.2)()
        swift_code = SwiftCode(font_normal, font_normal, marker_prob=1, down_prob=0)()
        bank(bank_name, bank_address, account_number, account_name, swift_code)

        fields = bank.get_fields()
        for field in fields:
            x1, y1, x2, y2 = field['box']
            bank.canvas = cv2.rectangle(bank.canvas, (x1, y1), (x2, y2), (0, 255, 0), thickness = 1)

        cv2.imshow('img', bank.canvas)
        key = cv2.waitKey(0)
        if key == ord('q'):
            exit(-1)