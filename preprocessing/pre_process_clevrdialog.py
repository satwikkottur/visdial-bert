"""Preprocess CLEVR-Dialog to run Visdial-BERT.

Author(s): Satwik Kottur
"""
import os
import concurrent.futures
import json
import argparse
import glob
import importlib
import sys
from pytorch_transformers.tokenization_bert import BertTokenizer

import torch


def read_options(argv=None):
    parser = argparse.ArgumentParser(description="Options")
    # -------------------------------------------------------------------------
    # Data input settings
    parser.add_argument(
        "-visdial_train",
        default="data/visdial/visdial_1.0_train.json",
        help="json file containing train split of visdial data",
    )

    parser.add_argument(
        "-visdial_val",
        default="data/visdial/visdial_1.0_val.json",
        help="json file containing val split of visdial data",
    )
    parser.add_argument(
        "-visdial_test",
        default="data/visdial/visdial_1.0_test.json",
        help="json file containing test split of visdial data",
    )

    parser.add_argument(
        "-max_seq_len",
        default=1024,
        type=int,
        help="the max len of the input representation of the dialog encoder",
    )
    # -------------------------------------------------------------------------
    # Logging settings

    parser.add_argument(
        "-save_path_train",
        default="data/visdial/visdial_1.0_train_processed.json",
        help="Path to save processed train json",
    )
    parser.add_argument(
        "-save_path_val",
        default="data/visdial/visdial_1.0_val_processed.json",
        help="Path to save val json",
    )
    parser.add_argument(
        "-save_path_test",
        default="data/visdial/visdial_1.0_test_processed.json",
        help="Path to save test json",
    )

    parser.add_argument(
        "-save_path_train_dense_samples",
        default="data/visdial/visdial_1.0_train_dense_processed.json",
        help="Path to save processed train json",
    )

    try:
        parsed = vars(parser.parse_args(args=argv))
    except IOError as msg:
        parser.error(str(msg))
    return parsed


if __name__ == "__main__":
    params = read_options()
    # read all the three splits
    with open(params["visdial_train"]) as file_id:
        input_train = json.load(file_id)
        input_train_data = input_train["data"]["dialogs"]
        train_questions = input_train["data"]["questions"]
        train_answers = input_train["data"]["answers"]

    with open(params["visdial_val"]) as file_id:
        input_val = json.load(file_id)
        input_val_data = input_val["data"]["dialogs"]
        val_questions = input_val["data"]["questions"]
        val_answers = input_val["data"]["answers"]

    f = open(params["visdial_test"])
    input_test = json.load(f)
    input_test_data = input_test["data"]["dialogs"]
    test_questions = input_test["data"]["questions"]
    test_answers = input_test["data"]["answers"]
    f.close()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    max_seq_len = params["max_seq_len"]
    num_illegal_train = 0
    num_illegal_val = 0
    num_illegal_test = 0
    # process train
    i = 0
    max_len = 0
    while i < len(input_train_data):
        if i % 1000 == 0:
            print(i)
        cur_dialog = input_train_data[i]["dialog"]
        caption = input_train_data[i]["caption"]
        tot_len = 22 + len(
            tokenizer.encode(caption)
        )  # account for 21 sep tokens, CLS token and caption
        for rnd in range(len(cur_dialog)):
            if "answer" in cur_dialog[rnd]:
                tot_len += len(
                    tokenizer.encode(train_answers[cur_dialog[rnd]["answer"]])
                ) + 1
                tot_len += len(
                    tokenizer.encode(train_questions[cur_dialog[rnd]["question"]])
                ) + 1
            else:
                tot_len += len(
                    tokenizer.encode(train_questions[cur_dialog[rnd]["caption"]])
                ) + 1

        max_len = max(max_len, tot_len)
        if tot_len > 512:
            print("Train: " + str(tot_len))
            num_illegal_train += 1
        i += 1
        # if tot_len <= max_seq_len:
        #     i += 1
        # else:
        #     input_train_data.pop(i)
        #     num_illegal_train += 1

    # # print(max_len)
    # train_img_id_to_index = {
    #     input_train_data[i]["image_id"]: i for i in range(len(input_train_data))
    # }
    # print(max_len)
    max_len = 0

    # process val
    i = 0
    num_illegal_val = 0
    while i < len(input_val_data):
        if i % 1000 == 0:
            print(i)
        cur_dialog = input_val_data[i]["dialog"]
        caption = input_val_data[i]["caption"]
        tot_len = 1  # CLS token
        tot_len += len(tokenizer.encode(caption)) + 1
        for rnd in range(len(cur_dialog)):
            if "answer" in cur_dialog[rnd]:
                tot_len += len(
                    tokenizer.encode(train_answers[cur_dialog[rnd]["answer"]])
                )
                tot_len += len(
                    tokenizer.encode(train_questions[cur_dialog[rnd]["question"]])
                )
            else:
                tot_len += len(
                    tokenizer.encode(train_questions[cur_dialog[rnd]["caption"]])
                )
        if tot_len > 512:
            print("Val: " + str(total_len))
            num_illegal_val += 1
        max_len = max(max_len, tot_len)
        i += 1

    i = 0
    num_illegal_test = 0
    # process test
    while i < len(input_test_data):
        if i % 1000 == 0:
            print(i)
        remove = False
        cur_dialog = input_test_data[i]["dialog"]
        input_test_data[i]["round_id"] = len(cur_dialog)
        caption = input_test_data[i]["caption"]
        tot_len = 1  # CLS token
        tot_len += len(tokenizer.encode(caption)) + 1
        for rnd in range(len(cur_dialog)):
            if "answer" in cur_dialog[rnd]:
                tot_len += len(
                    tokenizer.encode(train_answers[cur_dialog[rnd]["answer"]])
                ) + 1
                tot_len += len(
                    tokenizer.encode(train_questions[cur_dialog[rnd]["question"]])
                ) + 1
            else:
                tot_len += len(
                    tokenizer.encode(train_questions[cur_dialog[rnd]["caption"]])
                ) + 1

        max_len = max(max_len, tot_len)
        if tot_len > 512:
            num_illegal_test += 1
            print("Test: " + str(tot_len))
        max_len_cur_sample = tot_len
        i += 1
    print("Illegal (train): {}".format(num_illegal_train))
    print("Illegal (val): {}".format(num_illegal_val))
    print("Illegal (test): {}".format(num_illegal_test))
    print(max_len)
    import pdb; pdb.set_trace()
    """
    # store processed files
    """
    with open(params["save_path_train"], "w") as train_out_file:
        json.dump(input_train, train_out_file)

    with open(params["save_path_val"], "w") as val_out_file:
        json.dump(input_val, val_out_file)
    with open(params["save_path_val_ndcg"], "w") as val_ndcg_out_file:
        json.dump(input_val_ncdg, val_ndcg_out_file)
    with open(params["save_path_test"], "w") as test_out_file:
        json.dump(input_test, test_out_file)

    # spit stats
    print("number of illegal train samples", num_illegal_train)
    print("number of illegal val samples", num_illegal_val)
    print("number of illegal test samples", num_illegal_test)
