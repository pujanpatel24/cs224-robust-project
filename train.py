import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering, DistilBertForMaskedLM
from transformers import AdamW
from tensorboardX import SummaryWriter
from google_trans_new import google_translator
import time
import torch.nn as nn
import pdb

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from tqdm import tqdm

lamb = 5e-5

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st: offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples


def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    # TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split == 'train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples


# TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log, tokenizer):
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.mlm_probability = args.mlm_probability
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        self.tokenizer = tokenizer
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
             tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                data_loader.dataset.encodings,
                                                (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, model, train_dataloader, eval_dataloader, val_dict, patience):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)
        curr_patience = 0

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    # loss = outputs[0]
                    # regularization = torch.sum(torch.square(torch.argmax(outputs[1], dim=1) - torch.argmax(outputs[2], dim=1))) / outputs[1].shape[0]
                    # print(outputs[0])
                    # print(outputs[1].shape)
                    # print(torch.argmax(outputs[1], dim=1))
                    # print(torch.argmax(outputs[2], dim=1))
                    # print(regularization)

                    # loss_1 = nn.MSELoss()
                    # start_logits = torch.argmax(outputs[1], dim=1).float()
                    # start_logits.requires_grad = True
                    # end_logits = torch.argmax(outputs[2], dim=1).float()
                    # end_logits.requires_grad = True
                    # output2 = loss_1(start_logits, end_logits)
                    # loss = outputs[0] + lamb * output2
                    loss = outputs[0]
                    pdb.set_trace();
                    one_hot = torch.zeros(outputs[1].shape)
                    for i in range(one_hot_start.shape[0]):
                        one_hot[i][start_positions[i]] = 1
                        one_hot[i][end_positions[i]] = 1
                    loss2 = nn.CrossEntropyLoss()
                    # start_positions = start_positions.squeeze(-1)
                    # end_positions = end_positions.squeeze(-1)
                    # ignored_index = outputs[1].size(1)
                    # start_positions.clamp_(0, ignored_index)
                    # end_positions.clamp_(0, ignored_index)
                    outputs2 = loss2(outputs[1]+outputs[2], one_hot)
                    loss += lamb * outputs2
                    pdb.set_trace();
                    loss.backward()
                    optim.step()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                        else:
                            curr_patience += 1
                        if curr_patience == patience:
                            self.log.info(
                                f'Hit patience of {patience}. Early stopping at epoch {epoch_num} and step {global_idx}...')
                            return best_scores
                    global_idx += 1
        return best_scores

    # SOURCE: https://github.com/allenai/dont-stop-pretraining/blob/master/mlm_study/huggingface_study/mlm.py
    def _mask_samples(self, inputs):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def trainMLM(self, model, train_dataloader):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        tbx = SummaryWriter(self.save_dir)
        best_loss = float('inf')

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    attention_mask = batch['attention_mask'].to(device)
                    masked_inputs, labels = self._mask_samples(batch['input_ids'])
                    masked_inputs = masked_inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(masked_inputs, attention_mask=attention_mask, labels=labels)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
                    progress_bar.update(len(masked_inputs))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        if loss < best_loss:
                            best_loss = loss
                            self.log.info(f'Saving model at step: {epoch_num}')
                            self.save(model)
                    global_idx += 1


def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name = ''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
        if args.backtranslation and split_name == 'train':
            with open(data_dir + '_back/' + dataset + '.json', 'rb') as f:
                aug_dict = json.load(f)
                dataset_dict = util.merge(dataset_dict, aug_dict)
        if args.synonym and split_name == 'train':
            with open(data_dir + '_syn/' + dataset + '.json', 'rb') as f:
                aug_dict = json.load(f)
                dataset_dict = util.merge(dataset_dict, aug_dict)
        if split_name == 'val':
            if dataset == 'duorc':
                print("Before: ", dataset, len(dataset_dict['question']))
                dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
                dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
                print("After: ", dataset, len(dataset_dict['question']))
            elif dataset == 'relation_extraction':
                print("Before: ", dataset, len(dataset_dict['question']))
                for i in range(6):
                    dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
                print("After: ", dataset, len(dataset_dict['question']))
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return util.QADataset(data_encodings, train=(split_name == 'train')), dataset_dict


def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    model_path = args.model_path if args.model_path else 'distilbert-base-uncased'
    if args.pretrain_MLM:
        model = DistilBertForMaskedLM.from_pretrained(model_path)
    else:
        model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.pretrain_MLM:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data for MLM...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log, tokenizer)
        _, train_dict = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
        data_encodings = tokenizer(train_dict['context'],
                                   truncation=True,
                                   stride=128,
                                   max_length=384,
                                   padding='max_length')
        train_dataset = util.MLMDataset(data_encodings)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=RandomSampler(train_dataset))
        trainer.trainMLM(model, train_loader)

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log, tokenizer)
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict, args.patience)
        return best_scores
    # Cited from https://arxiv.org/pdf/2006.05987.pdf
    # if reinit_layers is > 0 then, we fine tune on OOD with reinitializing the top reinit_layers
    if args.do_finetune:
        print('Finetuning model on OOD')

        # # I'm not sure if we're supposed to freeze layers. I'm gonna wanna talk this over with someone.
        # for param in model.distilbert.parameters():
        #     param.requires_grad = False
        #     log.info("Freezing embedding layer and transformer blocks ")

        # This will be code for reinitializing the top num-layer transformer blocks
        for layer in range(args.num_layers):
            # Get top most layers
            layer = 5 - layer
            for module in model.distilbert.transformer.layer[layer].modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

        # Standard fine tuning approach of just fine-tuning the output layer after reinitializing.
        # We should probably look to see if we want to change this output layer.
        model.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        model.qa_outputs.bias.data.zero_()

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Finetuning Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log, tokenizer)
        finetune_dataset, _ = get_dataset(args, 'duorc,race,relation_extraction', 'datasets/oodomain_train', tokenizer,
                                          'train')
        log.info("Preparing Finetuning OOD Data...")
        finetune_val_dataset, finetune_val_dict = get_dataset(args, 'duorc,race,relation_extraction',
                                                              'datasets/oodomain_val', tokenizer, 'val')
        train_loader = DataLoader(finetune_dataset,
                                  batch_size=args.batch_size,
                                  sampler=RandomSampler(finetune_dataset))
        val_loader = DataLoader(finetune_val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(finetune_val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, finetune_val_dict, args.patience)
        return best_scores
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log, tokenizer)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
