#! /usr/bin/env python
"""
TODO: Write a description here!
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json


def main(args):
    with open(args["train_clevr_dialog_json"], "r") as file_id:
        train_data = json.load(file_id)

    num_instances = 20000
    train_data["data"]["dialogs"] = train_data["data"]["dialogs"][:num_instances]
    save_path = args["train_clevr_dialog_json"].replace(".json", "_20k.json")
    print("Saving: {}".format(save_path))
    with open(save_path, "w") as file_id:
        json.dump(train_data, file_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_clevr_dialog_json",
        default="-",
        help="CLEVR-Dialog train (VD format)"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
