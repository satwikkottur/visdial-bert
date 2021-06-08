import torch
from torch.utils import data
import json
from pytorch_transformers.tokenization_bert import BertTokenizer
import numpy as np
import random
from utils.data_utils import list2tensorpad, encode_input, encode_image_input
from utils.image_features_reader import ImageFeaturesH5Reader


class CLEVRDialogDataset(data.Dataset):
    def __init__(self, params):

        self.numDataPoints = {}
        num_samples_train = params["num_train_samples"]
        num_samples_val = params["num_val_samples"]
        self._image_features_reader = ImageFeaturesH5Reader(
            params["visdial_image_feats"]
        )
        with open(params["visdial_processed_train"]) as f:
            self.visdial_data_train = json.load(f)
            if params["overfit"]:
                if num_samples_train:
                    self.numDataPoints["train"] = num_samples_train
                else:
                    self.numDataPoints["train"] = 5
            else:
                if num_samples_train:
                    self.numDataPoints["train"] = num_samples_train
                else:
                    self.numDataPoints["train"] = len(
                        self.visdial_data_train["data"]["dialogs"]
                    )

        with open(params["visdial_processed_val"]) as f:
            self.visdial_data_val = json.load(f)
            if params["overfit"]:
                if num_samples_val:
                    self.numDataPoints["val"] = num_samples_val
                else:
                    self.numDataPoints["val"] = 5
            else:
                if num_samples_val:
                    self.numDataPoints["val"] = num_samples_val
                else:
                    self.numDataPoints["val"] = len(
                        self.visdial_data_val["data"]["dialogs"]
                    )
        with open(params["visdial_processed_test"]) as f:
            self.visdial_data_test = json.load(f)
            self.numDataPoints["test"] = len(self.visdial_data_test["data"]["dialogs"])

        self.overfit = params["overfit"]
        self.num_options = params["num_options"]
        self._split = "train"
        self.subsets = ["train", "val", "test"]
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = tokenizer
        # fetching token indicecs of [CLS] and [SEP]
        tokens = ["[CLS]", "[MASK]", "[SEP]"]
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        self.CLS = indexed_tokens[0]
        self.MASK = indexed_tokens[1]
        self.SEP = indexed_tokens[2]
        self.params = params
        self._max_region_num = 49 if self.params["deep_dialogs"] else 37

    def __len__(self):
        return self.numDataPoints[self._split]

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets
        self._split = split

    def __getitem__(self, index):
        def tokens2str(seq):
            dialog_sequence = ""
            for sentence in seq:
                for word in sentence:
                    dialog_sequence += self.tokenizer._convert_id_to_token(word) + " "
                dialog_sequence += " </end> "
            dialog_sequence = dialog_sequence.encode("utf8")
            return dialog_sequence

        def pruneRounds(context, num_rounds):
            start_segment = 1
            len_context = len(context)
            cur_rounds = (len(context) // 2) + 1
            l_index = 0
            if cur_rounds > num_rounds:
                # caption is not part of the final input
                l_index = len_context - (2 * num_rounds)
                start_segment = 0
            return context[l_index:], start_segment

        # Combining all the dialog rounds with the [SEP] and [CLS] token
        MAX_SEQ_LEN = self.params["max_seq_len"]
        cur_data = None
        if self._split == "train":
            cur_data = self.visdial_data_train["data"]
        elif self._split == "val":
            if self.overfit:
                cur_data = self.visdial_data_train["data"]
            else:
                cur_data = self.visdial_data_val["data"]
        else:
            cur_data = self.visdial_data_test["data"]

        # number of options to score on
        num_options = self.num_options
        assert num_options > 1 and num_options <= 100

        dialog = cur_data["dialogs"][index]
        cur_questions = cur_data["questions"]
        cur_answers = cur_data["answers"]
        img_id = dialog["image_id"]

        utterances = []
        tokenized_caption = self.tokenizer.encode(dialog["caption"])
        utterances.append([tokenized_caption])
        # add a 1 for the CLS token as well as the sep tokens which 
        # follows the caption
        tot_len = (
            len(tokenized_caption) + 2
        )
        answer_inds = []
        previous_caption = False
        for rnd, utterance in enumerate(dialog["dialog"]):
            if self.params["ignore_history"]:
                cur_rnd_utterance = []
            else:
                cur_rnd_utterance = utterances[-1].copy()
            # Remove the answer from the previous utterance.
            # Dont want to predict answer given answer!
            if rnd > 0 and not previous_caption:
                utterances[-1].pop(-1)

            # If previous was caption, remove from utterances.
            if previous_caption:
                assert self.params["deep_dialogs"], (
                    "Captions part of deep dialogs only!"
                )
                utterances.pop(-1)
                previous_caption = False

            if "caption" in utterance:
                tokenized_caption = self.tokenizer.encode(
                    cur_questions[utterance["caption"]]
                )
                cur_rnd_utterance.append(tokenized_caption)
                caption_len = len(tokenized_caption)
                # the additional 1 is for the sep token
                tot_len += caption_len + 1
                previous_caption = True
            else:
                tokenized_question = self.tokenizer.encode(
                    cur_questions[utterance["question"]]
                )
                tokenized_answer = self.tokenizer.encode(
                    cur_answers[utterance["answer"]]
                )
                answer_inds.append(utterance["answer"])
                cur_rnd_utterance.append(tokenized_question)
                cur_rnd_utterance.append(tokenized_answer)

                question_len = len(tokenized_question)
                answer_len = len(tokenized_answer)
                # the additional 1 is for the sep token
                tot_len += question_len + 1
                # the additional 1 is for the sep token
                tot_len += answer_len + 1

            # If history length > 512, start clipping from the start.
            while tot_len > 512:
                # Make it empty instead of removing.
                # removed_round = cur_rnd_utterance.pop(0)
                # tot_len -= len(removed_round)
                print("Clipped!")
                for index, ii in enumerate(cur_rnd_utterance):
                    if len(ii) > 0:
                        break
                tot_len -= len(ii)
                cur_rnd_utterance[index] = []

            # randomly select one random utterance in that round
            utterances.append(cur_rnd_utterance)

        # Remove the answer for the last round.
        utterances[-1].pop(-1)
        # removing the caption in the beginning
        utterances = utterances[1:]

        if self.params["deep_dialogs"]:
            assert len(utterances) == 30
        else:
            assert len(utterances) == 10

        tokens_all_rnd = []
        mask_all_rnd = []
        segments_all_rnd = []
        sep_indices_all_rnd = []
        next_labels_all_rnd = []
        hist_len_all_rnd = []

        for j, context in enumerate(utterances):
            tokens_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            next_labels_all = []
            hist_len_all = []

            context, start_segment = pruneRounds(
                context, self.params["visdial_tot_rounds"]
            )
            # print("{}: {}".format(j, tokens2str(context)))
            max_sep_len = 64 if self.params["deep_dialogs"] else 25
            tokens, segments, sep_indices, mask = encode_input(
                context,
                start_segment,
                self.CLS,
                self.SEP,
                self.MASK,
                max_seq_len=MAX_SEQ_LEN,
                max_sep_len=max_sep_len,
                mask_prob=self.params["mask_prob"],
            )
            tokens_all.append(tokens)
            mask_all.append(mask)
            sep_indices_all.append(sep_indices)
            next_labels_all.append(torch.LongTensor([0]))
            segments_all.append(segments)
            hist_len_all.append(torch.LongTensor([len(context) - 1]))

            tokens_all_rnd.append(torch.cat(tokens_all, 0).unsqueeze(0))
            mask_all_rnd.append(torch.cat(mask_all, 0).unsqueeze(0))
            segments_all_rnd.append(torch.cat(segments_all, 0).unsqueeze(0))
            sep_indices_all_rnd.append(torch.cat(sep_indices_all, 0).unsqueeze(0))
            next_labels_all_rnd.append(torch.cat(next_labels_all, 0).unsqueeze(0))
            hist_len_all_rnd.append(torch.cat(hist_len_all, 0).unsqueeze(0))

        tokens_all_rnd = torch.cat(tokens_all_rnd, 0)
        mask_all_rnd = torch.cat(mask_all_rnd, 0)
        segments_all_rnd = torch.cat(segments_all_rnd, 0)
        sep_indices_all_rnd = torch.cat(sep_indices_all_rnd, 0)
        next_labels_all_rnd = torch.cat(next_labels_all_rnd, 0)
        # NOTE: Change this to answer id for CLEVR-Dialog.
        next_labels_all_rnd = torch.LongTensor(answer_inds).view(-1, 1)
        hist_len_all_rnd = torch.cat(hist_len_all_rnd, 0)

        item = {}
        item["tokens"] = tokens_all_rnd
        item["segments"] = segments_all_rnd
        item["sep_indices"] = sep_indices_all_rnd
        item["mask"] = mask_all_rnd
        item["next_sentence_labels"] = next_labels_all_rnd
        item["hist_len"] = hist_len_all_rnd

        if self.params["deep_dialogs"]:
            # Get three image features.
            images = img_id.split("=")[0].split(":")
            joint = {"features":[] , "num_boxes": 0, "boxes": [], "target": []}
            for image in images:
                el_features, el_num_boxes, el_boxes, _, el_image_target = (
                    self._image_features_reader[image]
                )
                joint["features"].append(el_features)
                joint["num_boxes"] += el_num_boxes
                joint["boxes"].append(el_boxes)
                joint["target"].append(el_image_target)
            (
                features,
                spatials,
                image_mask,
                image_target,
                image_label,
            ) = encode_image_input(
                np.concatenate(joint["features"], axis=0),
                joint["num_boxes"],
                np.concatenate(joint["boxes"], axis=0),
                np.concatenate(joint["target"], axis=0),
                max_regions=self._max_region_num,
            )
        else:
            # Modify the image id to remove the dialog_id.
            img_id = img_id.rsplit("_", 1)[0]
            # get image features
            features, num_boxes, boxes, _, image_target = self._image_features_reader[
                img_id
            ]
            (
                features,
                spatials,
                image_mask,
                image_target,
                image_label,
            ) = encode_image_input(
                features,
                num_boxes,
                boxes,
                image_target,
                max_regions=self._max_region_num,
            )
        item["image_feat"] = features
        item["image_loc"] = spatials
        item["image_mask"] = image_mask
        item["image_target"] = image_target
        item["image_label"] = image_label
        return item
