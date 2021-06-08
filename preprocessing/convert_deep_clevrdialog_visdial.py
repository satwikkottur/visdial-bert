#! /usr/bin/env python
"""
TODO: Write a description here!
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import json
import os


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
    for dialog_datum in data:
        for round_datum in dialog_datum["data"][1:]:
            if "question" in round_datum:
                if round_datum["question"] not in questions:
                    questions[round_datum["question"]] = len(questions)
            else:
                if round_datum["caption"] not in questions:
                    questions[round_datum["caption"]] = len(questions)

    for dialog_datum in data:
        # image_filename = image_datum["image_filename"]
        dialog_turns = []
        for round_datum in dialog_datum["data"][1:]:
            if "question" in round_datum:
                new_turn = {
                    "question": questions[round_datum["question"]],
                    "answer": global_answers[str(round_datum["answer"])]
                }
            else:
                new_turn = {"caption": questions[round_datum["caption"]]}
            dialog_turns.append(new_turn)

        new_datum = {
            "image_id": "{}={}".format(
                ":".join(dialog_datum["image_filename"]),
                ":".join([str(ii) for ii in dialog_datum["dialog_index"]]),
            ),
            "caption": dialog_datum["data"][0]["caption"],
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

    # load val data.
    print("Reading: {}".format(args["val_clevr_dialog_json"]))
    with open(args["val_clevr_dialog_json"]) as file_id:
        val_data = json.load(file_id)

    # load test data.
    print("Reading: {}".format(args["test_clevr_dialog_json"]))
    with open(args["test_clevr_dialog_json"]) as file_id:
        test_data = json.load(file_id)

    # Collect all the answers to get global_answers.
    answers = {}
    for dialog_datum in train_data:
        for round_datum in dialog_datum["data"]:
            if "answer" not in round_datum:
                continue
            if str(round_datum["answer"]) not in answers:
                answers[str(round_datum["answer"])] = len(answers)
    global_answers = answers

    # Train split.
    save_path = os.path.join(args["save_root"], "deep_clevr_dialog_vd_train.json")
    convert_data_subset(train_data, global_answers, "train", save_path)

    # Val split.
    save_path = os.path.join(args["save_root"], "deep_clevr_dialog_vd_val.json")
    convert_data_subset(val_data, global_answers, "val", save_path)

    # Test split.
    save_path = os.path.join(args["save_root"], "deep_clevr_dialog_vd_test.json")
    convert_data_subset(test_data, global_answers, "test", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_clevr_dialog_json", default="-", help="Deep CLEVR-Dialog train"
    )
    parser.add_argument(
        "--val_clevr_dialog_json", default="-", help="Deep CLEVR-Dialog val"
    )
    parser.add_argument(
        "--test_clevr_dialog_json", default="-", help="Deep CLEVR-Dialog test"
    )
    parser.add_argument(
        "--save_root", default="-", help="Path to save the image features"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
