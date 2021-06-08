#! /usr/bin/env python
"""
TODO: Write a description here!
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import h5py
import json
import h5py
import os
import pdb
import numpy as np
import json
import sys
import csv
import base64
import pickle
import lmdb # install lmdb by "pip install lmdb"


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
IMG_HEIGHT = 320
IMG_WIDTH = 480
NUM_BOXES = 15
NUM_VAL_IMAGES = 500


def main(args):
    train_id = h5py.File(args["train_clevr_features"])
    val_id = h5py.File(args["val_clevr_features"])
    with open(args["train_clevr_map"]) as file_id:
        train_map = json.load(file_id)
    with open(args["val_clevr_map"]) as file_id:
        val_map = json.load(file_id)

    # Ensures that id and indexes are the same!
    for key, val in val_map["image_id_to_ix"].items():
        assert str(key) == str(val), "Something is wrong!"
    for key, val in train_map["image_id_to_ix"].items():
        assert str(key) == str(val), "Something is wrong!"

    # (x1, y1, x2, y2, width, height)
    # 480 × 320
    # 0, 2 -- width; 1, 3 -- height
    def create_feature_item(file_id, img_id, index):
        """Creates the feature id for CLEVR-Dialog.

        Args:
            file_id: h5 file id
            img_id: Label for the image
            index: Index in the file id

        Returns:
            item: dict of FIELDNAMES
        """
        spatial_feats = file_id["spatial_features"][index]
        spatial_feats[:, 0] *= IMG_WIDTH
        spatial_feats[:, 2] *= IMG_WIDTH
        spatial_feats[:, 1] *= IMG_HEIGHT
        spatial_feats[:, 3] *= IMG_HEIGHT
        boxes = np.ascontiguousarray(spatial_feats[:, :4])
        item = {
            "image_id": img_id,
            "image_w": IMG_WIDTH,
            "image_h": IMG_HEIGHT,
            "num_boxes": NUM_BOXES,
            "boxes": base64.b64encode(boxes),
            "features": base64.b64encode(file_id["image_features"][index]),
        }
        return item

    # Saving the CLEVR features for test split.
    id_list = []
    save_path = os.path.join(args["save_root"], "clevr_dialog_butd.lmdb")
    env = lmdb.open(save_path, map_size=1099511627776)
    print("Saving: {}".format(save_path))
    with env.begin(write=True) as txn:
        for index in range(len(val_map["image_id_to_ix"])):
            file_name = "CLEVR_val_{:06d}.png".format(index)
            img_id = file_name.encode()
            id_list.append(img_id)
            item = create_feature_item(val_id, img_id, index)
            txn.put(img_id, pickle.dumps(item))
            # Print progress.
            if len(id_list) % 1000 == 0:
                print(len(id_list))

        for index in range(len(train_map["image_id_to_ix"])):
            file_name = "CLEVR_train_{:06d}.png".format(index)
            img_id = file_name.encode()
            id_list.append(img_id)
            item = create_feature_item(train_id, img_id, index)
            txn.put(img_id, pickle.dumps(item))
            # Print progress.
            if len(id_list) % 1000 == 0:
                print(len(id_list))

        # Write down the image id list.
        txn.put('keys'.encode(), pickle.dumps(id_list))

    train_id.close()
    val_id.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_clevr_features", default="-", help="CLEVR train images"
    )
    parser.add_argument(
        "--train_clevr_map", default="-", help="CLEVR train images mapping"
    )
    parser.add_argument(
        "--val_clevr_features", default="-", help="CLEVR to val images"
    )
    parser.add_argument(
        "--val_clevr_map", default="-", help="CLEVR to val images mapping"
    )
    parser.add_argument(
        "--save_root", default="-", help="Path to save the image features"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
