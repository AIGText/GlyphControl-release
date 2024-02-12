import easyocr
import os
import argparse
from PIL import Image
import numpy as np
import Levenshtein as lev

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __repr__(self) -> str:
        return str(self.avg)

class OCR_EM_Counter(object):
    '''Computes and stores the OCR Exactly Match Accuracy.'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.ocr_acc_em = {}
        self.ocr_acc_em_rate = 0

    def add_text(self, text):
        if text not in self.ocr_acc_em:
            self.ocr_acc_em[text] = AverageMeter()
        
    def update(self, text, ocr_result):
        ocr_texts = [item[1] for item in ocr_result]
        self.ocr_acc_em[text].update(text in ocr_texts)
        self.ocr_acc_em_rate = sum([value.sum for value in self.ocr_acc_em.values()]) / sum([value.count for value in self.ocr_acc_em.values()])
    
    def __repr__(self) -> str:
        ocr_str = ",".join([f"{key}:{repr(value)}" for key, value in self.ocr_acc_em.items()])
        return f"OCR Accuracy is {ocr_str}.\nOCR EM Accuracy is {self.ocr_acc_em_rate}."
        # return f"OCR EM Accuracy is {self.ocr_acc_em_rate}."
    
class OCR_EM_without_capitalization_Counter(object):
    '''Computes and stores the OCR Exactly Match Accuracy.'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.ocr_acc_em = {}
        self.ocr_acc_em_rate = 0

    def add_text(self, text):
        if text not in self.ocr_acc_em:
            self.ocr_acc_em[text] = AverageMeter()
        
    def update(self, text, ocr_result):
        ocr_texts = [item[1].lower() for item in ocr_result]
        self.ocr_acc_em[text].update(text.lower() in ocr_texts)
        self.ocr_acc_em_rate = sum([value.sum for value in self.ocr_acc_em.values()]) / sum([value.count for value in self.ocr_acc_em.values()])
    
    def __repr__(self) -> str:
        ocr_str = ",".join([f"{key}:{repr(value)}" for key, value in self.ocr_acc_em.items()])
        return f"OCR without capitalization Accuracy is {ocr_str}.\nOCR EM without capitalization Accuracy is {self.ocr_acc_em_rate}."

class OCR_Levenshtein_Distance(object):
    '''Computes and stores the OCR Levenshtein Distance Accuracy.'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.ocr_lev = {}
        self.ocr_lev_avg = 0

    def add_text(self, text):
        if text not in self.ocr_lev:
            self.ocr_lev[text] = AverageMeter()
    
    def update(self, text, ocr_result):
        ocr_texts = [item[1] for item in ocr_result]
        lev_distance = [lev.distance(text, ocr_text) for ocr_text in ocr_texts]
        if lev_distance:
            self.ocr_lev[text].update(min(lev_distance))
            self.ocr_lev_avg = sum([value.sum for value in self.ocr_lev.values()]) / sum([value.count for value in self.ocr_lev.values()])

    def __repr__(self) -> str:
        return f"The Average Levenshtein Distance between Groundtruth and OCR result is {self.ocr_lev_avg}."
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default = "evaluate/images/stablediffusion_DrawText_Spelling_0.01_0.1_random", help='data file path')
    parser.add_argument('--num', type=int, default = 4, help='num per words')
    args = parser.parse_args()
    
    reader = easyocr.Reader(['en'])
    print(f"Evaluate on {args.path}.")
    ocr_em_counter = OCR_EM_Counter()
    ocr_em_wc_counter = OCR_EM_without_capitalization_Counter()
    ocr_lev = OCR_Levenshtein_Distance()
    for item in os.listdir(args.path):
        text = item
        path = os.path.join(args.path, item)
        ocr_em_counter.add_text(text)
        ocr_em_wc_counter.add_text(text)
        ocr_lev.add_text(text)
        for sub_item in [item for item in os.listdir(path) if ".png" in item][:args.num]:
            sub_path = os.path.join(path, sub_item)
            try:
                image = Image.open(sub_path)
            except:
                continue
            image_array = np.array(image)
            ocr_result = reader.readtext(image_array)
            ocr_em_counter.update(text, ocr_result)
            ocr_em_wc_counter.update(text, ocr_result)
            ocr_lev.update(text, ocr_result)

    print(ocr_em_counter)
    print(ocr_em_wc_counter)
    print(ocr_lev)