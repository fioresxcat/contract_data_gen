import os
import numpy as np
import unidecode

hint_test = -1

# raw_bank_names_path = os.getcwd() + "/content/en_bank_name.txt"
# raw_add_path = os.getcwd() + "/content/en_com_add.txt"
# raw_name_abbre_path = os.getcwd() + "/content/en_com_name_abrre.txt"
# raw_foreign_person_path = os.getcwd() + "/content/en_per_name.txt"

# raw_vn_unsign_add_path = os.getcwd() + "/content/vn_unsign_add.txt"
# raw_vn_com_name_path = os.getcwd() + "/content/vn_com_name.txt"
# raw_vn_per_name_path = os.getcwd() + "/content/vn_per_name.txt"

# position_path = os.getcwd() + "/content/all_position.txt"

resources_dir = '/data2/tungtx2/VNG-DataGeneration/resources'
raw_bank_names_path = resources_dir + '/raw_bank_names.txt'
raw_add_path = resources_dir + '/raw_foreign_address.txt'
raw_com_name_abbre_path = resources_dir + '/raw_foreign_company_names.txt'
raw_foreign_person_path = resources_dir + '/raw_foreign_person_names.txt'

raw_vn_unsign_add_path = resources_dir + '/raw_unsign_vietnamese_address.txt'
raw_vn_com_name_path = resources_dir + '/raw_vietnamese_company_foreign_form.txt'
raw_vn_per_name_path = resources_dir + '/raw_vietnamese_names.txt'

position_path = resources_dir + '/all_position.txt'

with open(raw_bank_names_path, "r", encoding="utf-8") as f:
    bank_names = [unidecode.unidecode(x[:-1]) for x in f.readlines(hint_test)]

with open(raw_add_path, "r", encoding="utf-8") as f:
    com_add = [unidecode.unidecode(x[3:-1]) for x in f.readlines(hint_test) if len(x[3:-1].split()) >= 2]

with open(raw_com_name_abbre_path, "r", encoding="utf-8") as f:
    com_name_abbre = [unidecode.unidecode(x[:-1]) for x in f.readlines(hint_test) if len(x[:-1].split()) >= 2]

with open(raw_foreign_person_path, "r", encoding="utf-8") as f:
    per_name = [unidecode.unidecode(x[:-1]) for x in f.readlines(hint_test)]

with open(raw_vn_unsign_add_path, "r", encoding="utf-8") as f:
    vn_unsign_add = [unidecode.unidecode(x[:-1]) for x in f.readlines(hint_test)]

with open(raw_vn_com_name_path, "r", encoding="utf-8") as f:
    vn_com_name = [unidecode.unidecode(x[:-1]) for x in f.readlines(hint_test)]

with open(raw_vn_per_name_path, "r", encoding="utf-8") as f:
    vn_per_name = [unidecode.unidecode(x[:-1]) for x in f.readlines(hint_test)]

with open(position_path, "r", encoding="utf-8") as f:
    data = f.readlines(hint_test)
    oneword = [line[:-1] for line in data if len(line.split()) == 1]
    twoword = [line[:-1] for line in data if len(line.split()) == 2]
    threeword = [line[:-1] for line in data if len(line.split()) == 3]

    pos = {
        1: oneword,
        2: twoword,
        3: threeword
    }

content = {
    "en_bank_name" : bank_names,
    "en_com_add" : com_add,
    "en_com_name_abrre" : com_name_abbre,
    "en_per_name" : per_name,
    "vn_unsign_add" : vn_unsign_add,
    "vn_com_name" : vn_com_name,
    "vn_per_name" : vn_per_name,
    "pos" : pos
}

if __name__ == "__main__":
    from numpy.random import choice
    print(choice(content["en_bank_name"]))
    print(choice(content["en_com_add"]))
    print(choice(content["en_com_name_abrre"]))
    print(choice(content["en_per_name"]))
    print(choice(content["vn_unsign_add"]))
    print(choice(content["vn_com_name"]))
    print(choice(content["vn_per_name"]))
    print(choice(content["pos"]))