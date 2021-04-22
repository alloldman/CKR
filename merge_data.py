import numpy as np
import os
import csv
from utils import decode_base64
def load_datas():
    base_dir = './'
    files = os.listdir(base_dir)
    tsv_fieldnames = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'cls','conf']
    label_dict = {}
    for file_ in files:
        with open(os.path.join(base_dir,file_),'r') as f:
            reader = csv.DictReader(f, delimiter='\t', fieldnames = tsv_fieldnames)
            for item in reader:
                image_id = item['image_id']
                try:
                    cls_ = np.frombuffer(decode_base64(item['cls']), dtype=np.float)
                except:
                    print(image_id)
                label_dict[image_id] = np.unique(cls_.astype(np.int)).tolist()
    return label_dict
if __name__ == '__main__':
    x = load_datas()
