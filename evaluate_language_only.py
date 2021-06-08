import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from dataloader.dataloader_clevrdialog import CLEVRDialogDataset
import options
from models.language_only_dialog_encoder import DialogEncoder
import torch.optim as optim
from utils.visualize import VisdomVisualize
import pprint
from time import gmtime, strftime
from timeit import default_timer as timer
from pytorch_transformers.optimization import AdamW
import os
from utils.visdial_metrics import SparseGTMetrics, NDCG, scores_to_ranks
from pytorch_transformers.tokenization_bert import BertTokenizer
from utils.data_utils import sequence_mask, batch_iter
from utils.optim_utils import WarmupLinearScheduleNonZero
import json
import logging
from train_language_only_baseline import forward


def eval_ai_generate(dataloader, params, eval_batch_size, dialog_encoder):
    dialog_encoder.eval()
    batch_idx = 0
    with torch.no_grad():
        # batch_size = 500 * (params["n_gpus"] / 8)
        # batch_size = min(
        #     [1, 2, 4, 5, 100, 1000, 200, 8, 10, 40, 50, 500, 20, 25, 250, 125],
        #     key=lambda x: abs(x - batch_size) if x <= batch_size else float("inf"),
        # )
        # batch_size = min(eval_batch_size, batch_size)
        batch_size = eval_batch_size
        print("batch size for evaluation", batch_size)
        outputs = []
        if params["overfit"]:
            batch_size = 100
        for epoch_id, _, batch in batch_iter(dataloader, params):
            if epoch_id == 1:
                break
            tokens = batch["tokens"]
            num_rounds = tokens.shape[1]
            num_options = tokens.shape[2]
            tokens = tokens.view(-1, tokens.shape[-1])
            segments = batch["segments"]
            segments = segments.view(-1, segments.shape[-1])
            sep_indices = batch["sep_indices"]
            sep_indices = sep_indices.view(-1, sep_indices.shape[-1])
            mask = batch["mask"]
            mask = mask.view(-1, mask.shape[-1])
            hist_len = batch["hist_len"]
            hist_len = hist_len.view(-1)

            gt_labels = batch["next_sentence_labels"].view(-1)

            # print(
            #     tokens.shape[0],
            #     segments.shape[0],
            #     sep_indices.shape[0],
            #     mask.shape[0],
            #     hist_len.shape[0],
            #     num_rounds * num_options * eval_batch_size
            # )
            assert (
                tokens.shape[0]
                == segments.shape[0]
                == sep_indices.shape[0]
                == mask.shape[0]
                == hist_len.shape[0]
                == num_rounds * num_options * eval_batch_size
            )
            assert (eval_batch_size * num_rounds * num_options) // batch_size == (
                eval_batch_size * num_rounds * num_options
            ) / batch_size
            for j in range((eval_batch_size * num_rounds * num_options) // batch_size):
                # create chunks of the original batch
                item = {}
                start = j * batch_size
                end = (j + 1) * batch_size
                item['tokens'] = tokens[start:end, :]
                item['segments'] = segments[start:end, :]
                item['sep_indices'] = sep_indices[start:end, :]
                item['mask'] = mask[start:end, :]
                item['hist_len'] = hist_len[start:end]
                item["gt_labels"] = gt_labels[start:end]

                _, _, _, nsp_scores = forward(
                    dialog_encoder,
                    item,
                    params,
                    output_nsp_scores=True,
                    evaluation=True,
                )
                # normalize nsp scores
                # NOTE: CLEVR-Dialog 29 output space.
                assert nsp_scores.shape[-1] == 29.
                model_matches = (
                    item["gt_labels"].view(-1) == nsp_scores.argmax(dim=-1).cpu()
                )
                outputs.append(model_matches)
            print("Eval: {}".format(batch_idx))
            batch_idx += 1

    dialog_encoder.train()
    print("tot eval batches", batch_idx)
    all_metrics = {"accuracy": torch.mean(torch.cat(outputs).float())}
    print(all_metrics)
    return all_metrics


if __name__ == "__main__":
    params = options.read_command_line()
    pprint.pprint(params)
    dataset = CLEVRDialogDataset(params)
    eval_batch_size = params["batch_size"]
    # eval_batch_size = params["batch_size"] // 2
    split = "test"
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=params["num_workers"],
        drop_last=False,
        pin_memory=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params["device"] = device
    dialog_encoder = DialogEncoder()

    if params["start_path"]:
        pretrained_dict = torch.load(params["start_path"])

        if "model_state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["model_state_dict"]

        model_dict = dialog_encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("number of keys transferred", len(pretrained_dict))
        assert len(pretrained_dict.keys()) > 0
        model_dict.update(pretrained_dict)
        dialog_encoder.load_state_dict(model_dict)

    dialog_encoder = nn.DataParallel(dialog_encoder)
    dialog_encoder.to(device)
    ranks_json = eval_ai_generate(dataloader, params, eval_batch_size, dialog_encoder)

    # json.dump(ranks_json, open(params["save_name"] + "_predictions.txt", "w"))
