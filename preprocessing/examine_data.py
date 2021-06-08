# path = "data/visdial/visdial_1.0_val.json"
# 
# import json
# with open(path, "r") as file_id:
#     val_data = json.load(file_id)

import h5py
path = "data/clevrdialog/clevr_butd/val.hdf5"

file_id = h5py.File(path)

# with open(path, "r") as file_id:
#     val_data = json.load(file_id)

# (x1, y1, x2, y2, width, height)
import pdb; pdb.set_trace()
