#! /usr/bin/env python
"""
TODO: Write a description here!
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json
import os


NUM_VAL_IMGS = 500


def convert_data_subset(data, global_answers, split, save_path):
    """Convert this subset data to json file

    Args:
        data: Data subsplit
        global_answers: Global answer pool across different splits
        split: Split version CLEVR-Dialog
        save_path: Path to save the JSON
    """
    questions = {}
    dialogs = []
    for image_datum in data:
        for dialog_id, dialog_datum in enumerate(image_datum["dialogs"]):
            for round_datum in dialog_datum["dialog"]:
                if round_datum["question"] not in questions:
                    questions[round_datum["question"]] = len(questions)

    for image_datum in data:
        image_filename = image_datum["image_filename"]
        for dialog_id, dialog_datum in enumerate(image_datum["dialogs"]):
            dialog_turns = [
                {
                    "answer": global_answers[str(ii["answer"])],
                    "question": questions[ii["question"]]
                }
                for ii in dialog_datum["dialog"]
            ]
            new_datum = {
                "image_id": "{}_{}".format(
                    image_datum["image_filename"], dialog_id
                ),
                "caption": dialog_datum["caption"],
                "dialog": dialog_turns,
            }
            dialogs.append(new_datum)

    sorted_questions = [
        ii[0] for ii in sorted(questions.items(), key=lambda x: x[1])
    ]
    sorted_answers = [
        ii[0] for ii in sorted(global_answers.items(), key=lambda x: x[1])
    ]
    split_data = {
        "version": "clevr_dialog",
        "split": split,
        "data": {
            "dialogs": dialogs,
            "answers": sorted_answers,
            "questions": sorted_questions,
        }
    }
    print("Saving: {}".format(save_path))
    with open(save_path, "w") as file_id:
        json.dump(split_data, file_id)


def main(args):
    # Debug.
    # with open("data/clevrdialog/clevr_train_raw_70k_light.json", "r") as file_id:
    #     data = json.load(file_id)
    # dict_keys(['version', 'data', 'split'])
    # dict_keys(['image_id', 'dialog', 'caption'])
    # dict_keys(['answer', 'gt_index', 'question'])
    # {"answers": answers, "questions": questions, "dialogs": None}
    # limits = {"train": (0, 3500), "val": (3500, 4000), "test": (4000, 5000)}
    # for inst_id, label in enumerate(("train", "val", "test")):

    # Load train data.
    print("Reading: {}".format(args["train_clevr_dialog_json"]))
    with open(args["train_clevr_dialog_json"]) as file_id:
        train_data = json.load(file_id)

    # Load val data.
    print("Reading: {}".format(args["val_clevr_dialog_json"]))
    with open(args["val_clevr_dialog_json"]) as file_id:
        val_data = json.load(file_id)

    # Collect all the answers to get global_answers.
    answers = {}
    for image_datum in train_data:
        for dialog_id, dialog_datum in enumerate(image_datum["dialogs"]):
            for round_datum in dialog_datum["dialog"]:
                if str(round_datum["answer"]) not in answers:
                    answers[str(round_datum["answer"])] = len(answers)
    global_answers = answers

    # Train split.
    save_path = os.path.join(args["save_root"], "clevr_dialog_vd_train.json")
    convert_data_subset(train_data[NUM_VAL_IMGS:], global_answers, "train", save_path)

    # Val split.
    save_path = os.path.join(args["save_root"], "clevr_dialog_vd_val.json")
    convert_data_subset(train_data[:NUM_VAL_IMGS], global_answers, "val", save_path)

    # Test split.
    save_path = os.path.join(args["save_root"], "clevr_dialog_vd_test.json")
    convert_data_subset(val_data, global_answers, "test", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_clevr_dialog_json", default="-", help="CLEVR-Dialog train"
    )
    parser.add_argument(
        "--val_clevr_dialog_json", default="-", help="CLEVR-Dialog val"
    )
    parser.add_argument(
        "--save_root", default="-", help="Path to save the image features"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
