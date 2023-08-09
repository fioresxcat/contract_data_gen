from SubModule import *
from common import *
import cv2 as cv

source_txt = os.getcwd() + "/content/ds_cty.txt"
with open(source_txt, "r", encoding="utf-8") as f:
    lines = f.readlines()
    cty_lst = [x[:-1] for x in lines if "CÔNG TY" in x and "Mã số thuế" not in x]
    dc_lst = [" ".join(x[:-1].split(":")[-1].replace(",", " ").split(" ")) for x in lines if "Địa chỉ" in x]

class Buyer(Module):
    def __init__(self, shape=(3000, 3000), canvas=None):
        super().__init__(shape, canvas)

    def __call__(self, submodules: list):
        for sub_module in submodules:
            self.__paste__(submodule=sub_module, position=(1200, 100))

class Contract_Name(SubModule):
    def __init__(self, shape=None, canvas=None, marker_prob=0.5, marker_font: ImageFont.truetype = None, 
                 content_font: ImageFont.truetype = None, markers=None, content=None, label=None):
        super().__init__(shape, canvas, marker_prob, marker_font, content_font, markers, content, label)    

if __name__ == "__main__":
    test = Contract_Name(shape = (100, 2000), marker_font=normal, content_font=normal, 
                         marker_prob=0.8, markers=["A", "B"], content=dc_lst, label="contract_name")

    test()

    # print(test.fields)

    # for point in test.fields:
    #     x1, y1, x2, y2 = point['box']
    #     test.canvas = cv2.rectangle(test.canvas, (x1, y1), (x2, y2), (0, 0, 255), thickness = 2)

    # cv.imshow("test", test.canvas)
    # cv.waitKey(0)

    buyer_test = Buyer(shape = (3000, 3000))
    buyer_test([test])
    for point in buyer_test.get_fields():
        x1, y1, x2, y2 = point['box']

        buyer_test.canvas = cv2.rectangle(buyer_test.canvas, (x1, y1), (x2, y2), (0, 0, 255), thickness = 2)
    cv.imshow("test", buyer_test.canvas)
    key = cv.waitKey(0)    