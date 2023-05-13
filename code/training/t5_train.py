"""
This is the main file from 07.04
It includes Gradient Accumulation, which had a small revision 13.04

Developed based on baseline
Requires:
    * t5_helper_methods.py (common for ENglish/Norwegian)
    * t5_dataset.py: contains Dataset and CollationClass
    * t5_evaluate.py: evaluates with ERRANT
"""
import pandas
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW, Adam
import numpy as np
import time
from datetime import timedelta
import tqdm
import wandb
import os
import csv
import torchmetrics
import transformers
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorWithPadding
from argparse import ArgumentParser
from collections import namedtuple

import t5_dataset
from t5_dataset import T5GecDataset, CollationClass
from t5_helper_methods import * # these 2 can be merged later
from t5_evaluate import * # these 2 can be merged later

def main():
    # Set up arguments.
    parser = ArgumentParser()
    parser.add_argument("--description", type=str, default="non-defined_t5") # important to change for each slurm!
    parser.add_argument("--pretrained_model", type=str, default="t5-base")
    parser.add_argument("--data_path", default="../datasets/CLC_FCE/clc_fce_small_")
    parser.add_argument("--xtra_data_path", default="no_extra_dataset") # extra data for training
    parser.add_argument("--batch_size", action="store", type=int, default=8)
    parser.add_argument("--lr", action="store", type=float, default=0.0003) # learning rate
    parser.add_argument("--lr_decay", type=float, default=0.95) # not in use?
    parser.add_argument("--epochs", action="store", type=int, default=1)
    parser.add_argument("--pred_file", action="store", type=str, default="inspection.txt")
    parser.add_argument("--temperature", action="store", type=float, default=1.0) # parameter for generation
    parser.add_argument("--casefold", action="store_true")
    parser.add_argument("--use_small_dataset", action="store_true")
    parser.add_argument("--generation_max_length", action="store", type=int, default=50)
    parser.add_argument("--dropout_rate", type=float, default=0.1) # also the default of t5s
    parser.add_argument("--testing", action="store_true") # as opposed to development evaluating

    # from now on "args" represents variables that can be changed by passing from terminal
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, "------->", getattr(args, arg))

    # accomodate for GPU-use if available
    device = set_device()

    # read data to pandas from tsv. also optionally merge multiple training datasets
    # training data: standard/small/big -- sorry a bit ugly!
    train_path = args.data_path+'train.tsv'
    if args.use_small_dataset:
        train_path = args.data_path + "train_small.tsv" # remember: eval is still the same
    train_df = pandas.read_csv(train_path, sep='\t', header=0)
    if args.xtra_data_path != "no_extra_dataset": # if we want to use more training data
        train_df = merge_datasets(train_df, args.xtra_data_path)
    print("\nThe shape of the training dataframe: ", train_df.shape, "\n")
    # evaluation data: dev/test
    if args.testing:
        eval_df = pandas.read_csv(args.data_path+'test.tsv', sep='\t', header=0)
    else:
        eval_df = pandas.read_csv(args.data_path+'dev.tsv', sep='\t', header=0)

    # initiate tokenizer (defines vocabulary, indexing and so on. also padding)
    # tokenizer works also when byt5
    # if using mt5: pip install datasets transformers[sentencepiece]
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning
    if "byt5" in args.pretrained_model: # this is a byt5-type model
        t5_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, model_max_length=512)
        # increase the max length for generation, because "token" is a different unit.
        args.generation_max_length = args.generation_max_length * 5
        # increase the max length for tokenizer (truncation), because "token" is a different unit.
        tokenizer_max_length = 512
    else:
        t5_tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model, model_max_length=512)
        tokenizer_max_length = 150

    # build datasets, spits out raw sentences, tokenization in DataLoader because batch
    train_data = T5GecDataset(
        train_df,
        args)
    eval_data = T5GecDataset(
        eval_df,
        args)
    train_data_size = len(train_data.source) # could be useful to have?

    # dataloaders, custom collate function that utilizes t5-tokenizer for batches
    train_loader = DataLoader(train_data, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=CollationClass(t5_tokenizer, tokenizer_max_length, device))
    eval_loader = DataLoader(eval_data, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=CollationClass(t5_tokenizer, tokenizer_max_length, device))

    # define model and other training params
    my_model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model, dropout_rate=args.dropout_rate).to(device) # dropout_rate=0.1 er default
    total_steps = int((train_data_size / args.batch_size) * args.epochs) # should be same as number of batches in the whole training
    warmup_steps = int(total_steps / 12)
    optimizer = AdamW(my_model.parameters(), lr=args.lr) # can add weight_decay
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # initiate WANDB
    wandb.init(
        entity = "matiasjentoft", # username on wandb-webpage
        project = "round5.3", # group some runs together, marked "rounds" in thesis
        name = args.description # this is the name of this exact run
    )

    start = time.time()
    # train, evaluate with ERRANT, write predictions to files
    intermediate_results = []
    print('\nTraining starting...\n')
    for epoch in range(args.epochs):
        print('Training, epoch number:',epoch+1)

        # train on train-set
        my_model.train()
        train_iterator = tqdm.tqdm(train_loader)

        # implementing Gradient Accumulation
        accumulating_value = 32 / args.batch_size # 32 is the de-facto BS we want
        counter = accumulating_value
      
        optimizer.zero_grad(set_to_none=True) # reset gradients

        for source, target in train_iterator:
            #forward-pass and get loss directly
            loss = my_model(
                input_ids=source.input_ids, 
                attention_mask=source.attention_mask, 
                labels=target.input_ids).loss.to(device)
            
            loss = loss / accumulating_value # divide because otherwise accumulated gradient too big
            loss.backward()

            if counter == 1: # we have saved gradients enough times
                optimizer.step() # with accumulated gradients?
                train_iterator.set_postfix_str(f"loss: {round(loss.item(),4)}")
                # Log some stuff from every training step to wandb
                wandb_logging(epoch, wandb, optimizer, loss, score_dict={}, between_epochs=False)
                counter = accumulating_value # reset counter
                optimizer.zero_grad(set_to_none=True) # reset gradients
            else:
                counter -= 1
            scheduler.step() # warmup calculated with batch_size from argparser

        # evaluate on eval-set
        # prediction file for inspection with source, target, predictions
        # scores include all kind of stuff
        my_model.eval()
        final_epoch = (epoch+1) == args.epochs # True if last round¨
        if final_epoch:
            print("This is the final epoch")
        scores, prediction_file = evaluate_errant(args, my_model, optimizer, t5_tokenizer, eval_loader, device, final_epoch) # returns dict
        print(f"\nResults from evaluating after epoch {epoch+1}:\n")
        print(scores)
        print()
        # another (now redundant) logging method, created before wandb was introduced
        intermediate_results.append((epoch+1, scores["f_half"]))

        # Log every epoch evaluation to wandb
        wandb_logging(epoch, wandb, optimizer, loss, score_dict=scores, between_epochs=True)

    # saving final evaluation results
    final_scores = scores # dict
    final_predictions = prediction_file
    print(f"Predictions written to file: {final_predictions}")

    sec_duration = time.time() - start 
    duration = str(timedelta(seconds = sec_duration))

    # create a file with info on this run: cmd-arguments and results
    create_runfile(args, final_scores, device, intermediate_results, duration)
    print("Runfile created at: ", args.description, ".txt")


if __name__ == '__main__':
    main()
