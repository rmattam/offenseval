from __future__ import absolute_import, division, print_function
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from reader import OffenseEvalData
from bert import convert_examples_to_features

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertPredict(object):

    def __init__(self, args):
        self.args = args

        if self.args.local_rank == -1 or self.args.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}".format(
            self.device, n_gpu, bool(self.args.local_rank != -1)))

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)

        processor = OffenseEvalData()
        self.label_list = processor.get_labels()
        self.num_labels = len(self.label_list)

        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=True)

        # Load a trained model and config that you have fine-tuned
        output_model_file = os.path.join(self.args.bert_model_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(self.args.bert_model_dir, CONFIG_NAME)
        config = BertConfig(output_config_file)
        self.model = BertForSequenceClassification(config, num_labels=self.num_labels)
        self.model.load_state_dict(torch.load(output_model_file))
        self.model.to(self.device)
        self.model.eval()
        self.label_map = {i: label for i, label in enumerate(self.label_list)}

    def predict_one(self, test_input):

        eval_examples = [test_input]
        eval_features = convert_examples_to_features(
            eval_examples, self.label_list, self.args.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, labels=None)

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        return self.label_map[preds[0]]

