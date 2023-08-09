import sys
import os
sys.path.append(os.getcwd())
from sub_modules.sub_modules import *
from Module import Module
import numpy as np
from common import *

'''
All submodule: company_name, company_address, phone, fax, tax, 
reprenented_name, bank_name, bank_address, account_number,
account_name, swift_code 
'''

class Party(Module):
    def __init__(self, skip_prob:float = 0.3, down_prob:float = 0.7, bank_prob = 0.7, font_size = None, order_type='up|down'):
        self.skip_prob = skip_prob
        self.down_prob = down_prob
        self.bank_prob = bank_prob
        self.order_type = order_type
        self.font_size = font_size
        self.step_down = np.random.randint(7, 16)
        
        font = Font(font_scale=self.font_size)
        self.font_normal, self.font_bold, self.font_italic = font.get_font('normal'), font.get_font('bold'), font.get_font('italic')
        super().__init__()

        self.initialize_components()    

    def rand_font(self, fbold_prob=0.1, fitalic_prob=0.05):
        return np.random.choice([self.font_normal, self.font_bold, self.font_italic], p=[1-fbold_prob-fitalic_prob, fbold_prob, fitalic_prob])
                                
    def initialize_components(self):
        
        fbold_prob = 0.1
        fitalic_prob = 0.05
        def rand_font(bold_prob=fbold_prob, italic_prob=fitalic_prob):
            return np.random.choice([self.font_normal, self.font_bold, self.font_italic], p=[1-fbold_prob-fitalic_prob, fbold_prob, fitalic_prob])

        # company name
        company_name = CompanyName(rand_font(bold_prob=0.7), rand_font(bold_prob=0.9), marker_prob=0.8, down_prob=0.2)
        company_name.module_order_type = self.order_type
        company_name.random_capitalize_prob = {
            'marker': [0.2, 0.7, 0.1],  # marker cua buyer/seller ra it khi khong viet hoa
            'content': [0.1, 0.8, 0.1]  # content cua buyer/seller ra it khi khong viet hoa
        }
        company_name()

        company_address = Company_Address(rand_font(), rand_font(),marker_prob=0.7, down_prob=0.2)
        company_address.module_order_type = self.order_type
        company_address.random_capitalize_prob = {
            'marker': [0.4, 0.5, 0.1],
            'content': [0.4, 0.5, 0.1]
        }
        company_address()

        phone = Phone(rand_font(), rand_font(),marker_prob=1, down_prob = 0)()
        fax = Fax(rand_font(), rand_font(), marker_prob=1, down_prob = 0)()
        tax = Tax(rand_font(), rand_font(), marker_prob=1, down_prob = 0)()

        represented_name = RepresentedBy(rand_font(), rand_font(bold_prob=0.3), marker_prob=1, down_prob=0.0)
        represented_name.random_capitalize_prob = {
            'marker': [0.7, 0.3, 0],  
            'content': [0.7, 0.3, 0]  
        }
        represented_name()

        represented_position = RepresentedPosition(rand_font(), rand_font(), marker_prob=0.3, down_prob=0.0)
        represented_position.random_capitalize_prob = {
            'marker': [0.7, 0.3, 0],
            'content': [0.7, 0.3, 0]
        }
        represented_position()

        bank_name = BankName(rand_font(), rand_font(),marker_prob=0.7, down_prob=0)
        bank_name.module_order_type = self.order_type
        bank_name.random_capitalize_prob = {
            'marker': [0.5, 0.5, 0],  
            'content': [0.2, 0.8, 0] 
        }
        bank_name()

        bank_address = Bank_Address(rand_font(), rand_font(),marker_prob=0.7, down_prob=0.2)
        bank_address.module_order_type = self.order_type
        bank_address()

        account_number = AccountNumber(rand_font(), rand_font(), marker_prob=1)
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

        self.list_submodules = [company_name, company_address, phone, fax, tax, represented_name, represented_position, bank_name, bank_address, account_number, account_name, swift_code]


    def update_cursor(self, cursor, submodule, direction='down'):
        h, w = submodule.get_shape()
        if direction == 'down':
            # cursor[0] = cursor[0] #max(2, cursor[0] + randint(-5, 5))
            cursor[1] = cursor[1] + h + self.step_down + np.random.randint(-2, 2) #x1, y2
        elif direction == 'reset_down':
            cursor[0] = 10 #randint(5, 15)
            cursor[1] = cursor[1] + h + self.step_down + np.random.randint(-2, 2) #x1, y2
        else:
            cursor[0] = cursor[0] + w + randint(20, 100)
            # cursor[1] = cursor[1] #max(cursor[1] + randint(-5, 5), 2) #x2, y1
        return cursor
        

    def __call__(self):
        company_name, company_address, phone, fax, tax, represented_name, represented_position, \
                bank_name, bank_address, account_number, account_name, swift_code = self.list_submodules
        
        is_at_right = False
        #Paste company name and company_address_first
        cursor = [10, 10] # x1, y1, x2, y2
        self.__paste__(company_name, position=cursor)
        cursor = self.update_cursor(cursor, company_name, 'down')

        self.__paste__(company_address, position=cursor)
        cursor = self.update_cursor(cursor, company_address, 'reset_down')

        if np.random.rand() < 0.5:  # paste thong tin tel fax truoc bank

            # phone fax info
            list_submodules = [phone, fax, tax, represented_name]
            np.random.shuffle(list_submodules)
            idx = list_submodules.index(represented_name)
            # insert represented_position after represented_name
            list_submodules.insert(idx+1, represented_position)
            
            is_skip_representation = False
            for i, submodule in enumerate(list_submodules):
                if submodule == represented_position and is_skip_representation:
                    continue
                if np.random.random() < self.skip_prob: # skip component
                    if submodule == represented_name:
                        is_skip_representation = True
                    continue
                
                self.__paste__(submodule, position=cursor)

                if i != len(list_submodules) - 1:
                    if submodule == represented_name and represented_position.has_marker == False:
                        cursor = self.update_cursor(cursor, submodule, 'right')
                        is_at_right = True
                    elif submodule == represented_name and represented_position.has_marker == True:
                        cursor = self.update_cursor(cursor, submodule, 'down')
                        is_at_right = False

                    elif is_at_right: #next component must be down line
                        cursor = self.update_cursor(cursor, submodule, 'reset_down')
                        is_at_right = False

                    else:
                        if np.random.random() < self.down_prob: # Still downline
                            cursor = self.update_cursor(cursor, submodule, 'down')
                            is_at_right = False
                        else:
                            cursor = self.update_cursor(cursor, submodule, 'right')
                            is_at_right = True
                    
            
            # Bank info
            if np.random.random() < self.bank_prob:
                cursor = self.update_cursor(cursor, submodule, 'reset_down')  # reset cursor

                type = np.random.choice([1, 2], p=[0.6, 0.4]) if bank_name.has_marker else 2  # neu bank_name ko co marker, no phai nam cung dong voi account_name hoac nam ngay duoi
                if self.order_type == 'up|down':
                    bank_skip_prob_dict = {
                        'BankName': 0,
                        'Bank_Address': 0.4,
                        'AccountNumber': 0,
                        'AccountName': 0.6,
                        'SwiftCode': 0.2,
                        'Company_Address': 0.7,
                    }
                else:
                    bank_skip_prob_dict = {
                        'BankName': 0,
                        'Bank_Address': 0.5,
                        'AccountNumber': 0,
                        'AccountName': 0.8,
                        'SwiftCode': 0.2,
                        'Company_Address': 0.8,
                    }

                # build submodules list
                flag = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
                ls_submodules = []
                if flag == 1:
                    ls_submodules = [bank_name, bank_address, account_number, swift_code]
                elif flag == 2:
                    ls_submodules = [bank_name, bank_address, swift_code, account_number]
                elif flag == 3:
                    ls_submodules = [bank_name, account_number, swift_code, bank_address]
                elif flag == 4:
                    ls_submodules = [bank_name, swift_code, account_number, bank_address]
                elif flag == 5:
                    ls_submodules = [bank_name, swift_code, bank_address, account_number]
                elif flag == 6:
                    ls_submodules = [bank_name, account_number, bank_address, swift_code]


                # skip some submodules
                ls_idx2del = []
                for i, submodule in enumerate(ls_submodules):
                    if np.random.rand() < bank_skip_prob_dict[submodule.__class__.__name__]:
                        ls_idx2del.append(i)
                ls_submodules = [submodule for i, submodule in enumerate(ls_submodules) if i not in ls_idx2del]

                # chuyenr account number len dau
                if np.random.rand() < 0.25:  
                    ls_submodules.remove(account_number)                
                    ls_submodules = [account_number] + ls_submodules
                


                if type == 1: # moi field 1 dong
                    for submodule in ls_submodules:
                        self.__paste__(submodule, position=cursor)
                        cursor = self.update_cursor(cursor, submodule, 'down')

                elif type == 2: # co the co 2 field 1 dong
                    # dinh nghia cac submodule co the o cung 1 dong
                    ls_submodule_name_pair = [('AccountNumber', 'BankName'), ('AccountName', 'AccountNumber'), ('AccountNumber', 'SwiftCode')]
                    if bank_address.has_marker:
                        ls_submodule_name_pair.append(('BankName, Bank_Address'))
                    if len(ls_submodule_name_pair) == 4:
                        idx = np.random.choice([0, 1, 2, 3], p=[0.5, 0.15, 0.15, 0.2])
                        submodule_name_pair = ls_submodule_name_pair[idx]
                    else:
                        idx = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])
                        submodule_name_pair = ls_submodule_name_pair[idx]

                    submodule_pair = [submodule for submodule in ls_submodules if submodule.__class__.__name__ in submodule_name_pair]
                    if len(submodule_pair) >= 2:
                        np.random.shuffle(submodule_pair)
                        # neu bank_name ko co marker => no phai o sau account_number
                        if bank_name in submodule_pair and account_number in submodule_pair and (not bank_name.has_marker) and bank_name != submodule_pair[1]:
                            submodule_pair = submodule_pair[::-1]
                        # bank address phai nam sau bank name
                        elif bank_name in submodule_pair and bank_address in submodule_pair and bank_address != submodule_pair[1]:
                            submodule_pair = submodule_pair[::-1]
                            

                    is_submodule_pair_pasted = False
                    for submodule in ls_submodules:
                        if submodule not in submodule_pair:
                            self.__paste__(submodule, position=cursor)
                            cursor = self.update_cursor(cursor, submodule, 'down')
                        else:
                            if not is_submodule_pair_pasted:
                                for i, submodule in enumerate(submodule_pair):
                                    self.__paste__(submodule, position=cursor)
                                    update_cursor_type = 'reset_down' if i==len(submodule_pair)-1 else 'right'
                                    cursor = self.update_cursor(cursor, submodule, update_cursor_type)
                                is_submodule_pair_pasted = True
        else:
            # Bank info
            if np.random.random() < self.bank_prob:
                # cursor = self.update_cursor(cursor, submodule, 'reset_down')  # reset cursor

                type = np.random.choice([1, 2], p=[0.6, 0.4]) if bank_name.has_marker else 2  # neu bank_name ko co marker, no phai nam cung dong voi account_name hoac nam ngay duoi
                if self.order_type == 'up|down':
                    bank_skip_prob_dict = {
                        'BankName': 0,
                        'Bank_Address': 0.4,
                        'AccountNumber': 0,
                        'AccountName': 0.6,
                        'SwiftCode': 0.2,
                        'Company_Address': 0.7,
                    }
                else:
                    bank_skip_prob_dict = {
                        'BankName': 0,
                        'Bank_Address': 0.5,
                        'AccountNumber': 0,
                        'AccountName': 0.8,
                        'SwiftCode': 0.2,
                        'Company_Address': 0.8,
                    }

                # build submodules list
                flag = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
                ls_submodules = []
                if flag == 1:
                    ls_submodules = [bank_name, bank_address, account_number, swift_code]
                elif flag == 2:
                    ls_submodules = [bank_name, bank_address, swift_code, account_number]
                elif flag == 3:
                    ls_submodules = [bank_name, account_number, swift_code, bank_address]
                elif flag == 4:
                    ls_submodules = [bank_name, swift_code, account_number, bank_address]
                elif flag == 5:
                    ls_submodules = [bank_name, swift_code, bank_address, account_number]
                elif flag == 6:
                    ls_submodules = [bank_name, account_number, bank_address, swift_code]

                # skip some submodules
                ls_idx2del = []
                for i, submodule in enumerate(ls_submodules):
                    if np.random.rand() < bank_skip_prob_dict[submodule.__class__.__name__]:
                        ls_idx2del.append(i)
                ls_submodules = [submodule for i, submodule in enumerate(ls_submodules) if i not in ls_idx2del]

                # chuyenr account number len dau
                if np.random.rand() < 0.25:  
                    ls_submodules.remove(account_number)                
                    ls_submodules = [account_number] + ls_submodules
           


                if type == 1: # moi field 1 dong
                    for submodule in ls_submodules:
                        self.__paste__(submodule, position=cursor)
                        cursor = self.update_cursor(cursor, submodule, 'down')

                elif type == 2: # co the co 2 field 1 dong
                    # dinh nghia cac submodule co the o cung 1 dong
                    ls_submodule_name_pair = [('AccountNumber', 'BankName'), ('AccountName', 'AccountNumber'), ('AccountNumber', 'SwiftCode')]
                    if bank_address.has_marker:
                        ls_submodule_name_pair.append(('BankName, Bank_Address'))
                    if len(ls_submodule_name_pair) == 4:
                        idx = np.random.choice([0, 1, 2, 3], p=[0.5, 0.15, 0.15, 0.2])
                        submodule_name_pair = ls_submodule_name_pair[idx]
                    else:
                        idx = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])
                        submodule_name_pair = ls_submodule_name_pair[idx]

                    submodule_pair = [submodule for submodule in ls_submodules if submodule.__class__.__name__ in submodule_name_pair]
                    if len(submodule_pair) >= 2:
                        np.random.shuffle(submodule_pair)
                        # neu bank_name ko co marker => no phai o sau account_number
                        if bank_name in submodule_pair and account_number in submodule_pair and (not bank_name.has_marker) and bank_name != submodule_pair[1]:
                            submodule_pair = submodule_pair[::-1]
                        # bank address phai nam sau bank name
                        elif bank_name in submodule_pair and bank_address in submodule_pair and bank_address != submodule_pair[1]:
                            submodule_pair = submodule_pair[::-1]
                            

                    is_submodule_pair_pasted = False
                    for submodule in ls_submodules:
                        if submodule not in submodule_pair:
                            self.__paste__(submodule, position=cursor)
                            cursor = self.update_cursor(cursor, submodule, 'down')
                        else:
                            if not is_submodule_pair_pasted:
                                for i, submodule in enumerate(submodule_pair):
                                    self.__paste__(submodule, position=cursor)
                                    update_cursor_type = 'reset_down' if i==len(submodule_pair)-1 else 'right'
                                    cursor = self.update_cursor(cursor, submodule, update_cursor_type)
                                is_submodule_pair_pasted = True
                
            # phone fax info    
            list_submodules = [phone, fax, tax, represented_name]
            np.random.shuffle(list_submodules)
            idx = list_submodules.index(represented_name)
            # insert represented_position after represented_name
            list_submodules.insert(idx+1, represented_position)
            
            #other info of company
            is_skip_representation = False
            for i, submodule in enumerate(list_submodules):
                if submodule == represented_position and is_skip_representation:
                    continue
                if np.random.random() < self.skip_prob: # skip component
                    if submodule == represented_name:
                        is_skip_representation = True
                    continue
                
                self.__paste__(submodule, position=cursor)

                if i!= len(list_submodules) - 1:
                    if submodule == represented_name and represented_position.has_marker == False:
                        cursor = self.update_cursor(cursor, submodule, 'right')
                        is_at_right = True
                    elif submodule == represented_name and represented_position.has_marker == True:
                        cursor = self.update_cursor(cursor, submodule, 'down')
                        is_at_right = False

                    elif is_at_right: #next component must be down line
                        cursor = self.update_cursor(cursor, submodule, 'reset_down')
                        is_at_right = False

                    else:
                        if np.random.random() < self.down_prob: # Still downline
                            cursor = self.update_cursor(cursor, submodule, 'down')
                            is_at_right = False
                        else:
                            cursor = self.update_cursor(cursor, submodule, 'right')
                            is_at_right = True
    
        ## Draw now
        max_box = np.max([sub['box'] for sub in self.submodules], 0)
        xmax, ymax = max_box[2] + 5, max_box[3] + 5
        self.canvas = np.full((ymax, xmax, 3), 255, dtype=np.uint8)
        for i, sub in enumerate(self.submodules):
            x1, y1, x2, y2 = sub['box']
            self.canvas[y1:y2, x1:x2] = sub['submodule'].canvas

        return self
            

if __name__ == '__main__':
    from sub_modules.sub_modules import *
    # import cv2
    # for i in range(10):
    #     buyer = init_party(20)
    
    #     fields = buyer.get_fields()
    #     for field in fields:
    #         x1, y1, x2, y2 = field['box']
    #         buyer.canvas = cv2.rectangle(buyer.canvas, (x1, y1), (x2, y2), (0, 255, 0), thickness = 1)

    #     cv2.imshow('img', buyer.canvas)
    #     key = cv2.waitKey(0)
    #     if key == ord('q'):
    #         exit(-1)


    font = Font(font_scale=12)
    font_normal, font_bold, font_italic = font.get_font('normal'), font.get_font('bold'), font.get_font('italic')
    def rand_font():
        return np.random.choice([font_normal, font_bold, font_italic])

    bank_name = BankName(rand_font(), rand_font(),marker_prob=0.7, down_prob=0)
    bank_name()
    Image.fromarray(bank_name.canvas).save('a.jpg')
    ls_bank_name = [bank_name]    
    print(bank_name == ls_bank_name[0])

    bank_name()
    Image.fromarray(bank_name.canvas).save('a1.jpg')

    print(bank_name == ls_bank_name[0])