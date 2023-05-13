"""

Implements Gradient Accumulation
Adapted for Norwegian ASK corpus

Developed based on baseline
Requires:
    * t5_helper_methods.py (common for English/Norwegian)
    * norw_t5_dataset.py: contains Dataset and CollationClass
    * norw_t5_evaluate.py: evaluates with ERRANT
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
import transformers
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorWithPadding
from argparse import ArgumentParser
from collections import namedtuple # not in use?
from modeling_nort5 import NorT5ForConditionalGeneration

import norw_t5_dataset
from norw_t5_dataset import NorwT5GecDataset, CollationClass
from t5_helper_methods import * # common for English/Norwegian
from norw_t5_evaluate import * # adjusted for two evaluation datasets

def main():
    # Set up arguments.
    parser = ArgumentParser()
    parser.add_argument("--description", type=str, default="non-defined_t5") # important to change for each slurm!
    parser.add_argument("--pretrained_model", type=str, default="t5-base")
    parser.add_argument("--data_path", default="../datasets/ASK/")
    parser.add_argument("--ask_dataset", default="raw") # options: raw, expanded, both
    parser.add_argument("--batch_size", action="store", type=int, default=8)
    parser.add_argument("--lr", action="store", type=float, default=0.0003) # learning rate
    parser.add_argument("--epochs", action="store", type=int, default=1)
    parser.add_argument("--temperature", action="store", type=float, default=1.0) # parameter for generation
    parser.add_argument("--casefold", action="store_true")
    parser.add_argument("--generation_max_length", action="store", type=int, default=50)
    parser.add_argument("--dropout_rate", type=float, default=0.1) # also the default of t5s
    parser.add_argument("--testing", action="store_true")

    # from now on "args" represents variables that can be changed by passing from terminal
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, "------->", getattr(args, arg))

    # accomodate for GPU-use if available
    device = set_device()

    # read data to pandas from tsv. also optionally merge multiple training datasets
    # ASK: read both styles of dataset into Dataframe
    # TRAINING data
    train_dfs = []
    if args.ask_dataset in ["raw", "both"]:
        path = args.data_path+'ask_raw_train.tsv'
        train_dfs.append(pandas.read_csv(path, sep='\t', header=0, on_bad_lines="skip"))
    if args.ask_dataset in ["expanded", "both"]:
        path = args.data_path+'ask_exp_train.tsv'
        train_dfs.append(pandas.read_csv(path, sep='\t', header=0, on_bad_lines="skip"))
    if len(train_dfs) > 1:
        train_df = pandas.concat(train_dfs, ignore_index=True) # takes list as argument
    else:
        train_df = train_dfs[0]
    print(f"\nShape of the train dataset: ", train_df.shape)

    # EVALUATION data
    if args.testing: # final testing
        eval_raw_df = pandas.read_csv(args.data_path+'ask_raw_test.tsv', sep='\t', header=0, on_bad_lines="skip")
        eval_expanded_df = pandas.read_csv(args.data_path+'ask_exp_test.tsv', sep='\t', header=0, on_bad_lines="skip")
    else: # development
        eval_raw_df = pandas.read_csv(args.data_path+'ask_raw_dev.tsv', sep='\t', header=0, on_bad_lines="skip")
        eval_expanded_df = pandas.read_csv(args.data_path+'ask_exp_dev.tsv', sep='\t', header=0, on_bad_lines="skip")
    print(f"\nShape of the raw eval dataset: ", eval_raw_df.shape)
    print(f"\nShape of the exp eval dataset: ", eval_expanded_df.shape)
    print()


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
    elif "nort5" in args.pretrained_model:
        t5_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, model_max_length=512)
        tokenizer_max_length = 150
    else:
        t5_tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model, model_max_length=512)
        tokenizer_max_length = 150

    # build datasets, spits out raw sentences, tokenization in DataLoader because batch
    train_data = NorwT5GecDataset(
        train_df,
        args)
    eval_data_raw = NorwT5GecDataset(
        eval_raw_df,
        args)
    eval_data_expanded = NorwT5GecDataset(
        eval_expanded_df,
        args)
    train_data_size = len(train_data.source) # could be useful to have?

    # dataloaders, custom collate function that utilizes t5-tokenizer for batches
    train_loader = DataLoader(train_data, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=CollationClass(t5_tokenizer, tokenizer_max_length, device))
    eval_raw_loader = DataLoader(eval_data_raw, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=CollationClass(t5_tokenizer, tokenizer_max_length, device))
    eval_expanded_loader = DataLoader(eval_data_expanded, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=CollationClass(t5_tokenizer, tokenizer_max_length, device))

    # define model and other training params
    if "nort5" in args.pretrained_model:
        my_model = NorT5ForConditionalGeneration.from_pretrained(args.pretrained_model).to(device)
    else:
        my_model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model, dropout_rate=args.dropout_rate).to(device) # dropout_rate=0.1 er default
    total_steps = int((train_data_size / args.batch_size) * args.epochs) # should be same as number of batches in the whole training
    warmup_steps = int(total_steps / 12)
    optimizer = AdamW(my_model.parameters(), lr=args.lr) # can add weight_decay
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # initiate WANDB
    wandb.init(
        entity = "matiasjentoft", # username on wandb-webpage
        project = "round7-norw", # group some runs together, marked "rounds" in thesis
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

        for source, target, ec in train_iterator: # ErrorCode, "dummy" for RAW
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
                wandb_logging_norw(epoch, wandb, optimizer, loss, score_dict_list={}, between_epochs=False)
                counter = accumulating_value # reset counter
                optimizer.zero_grad(set_to_none=True) # reset gradients
            else:
                counter -= 1
            scheduler.step() # warmup calculated with batch_size from argparser

        # evaluate on dev-set
        # prediction file for inspection with source, target, predictions
        # scores include all kind of stuff
        my_model.eval()
        final_epoch = (epoch+1) == args.epochs # True if last round¨
        if final_epoch:
            print("This is the final epoch")
        both_errant, both_files = [], [] # [raw, exp], custom evaluation only EXPANDED
        dataset_styles = ["raw", "exp"] # used for iteration on out-filenames
        for i, eval_loader in enumerate([eval_raw_loader, eval_expanded_loader]):
            errant_dict, prediction_file, custom_dict = evaluate_errant(args, my_model, optimizer, t5_tokenizer, eval_loader, device, final_epoch, dataset_styles[i]) # returns dict
            intermediate_results.append((epoch+1, errant_dict["f_half"])) # basically irrelevant
            both_errant.append(errant_dict)
            both_files.append(prediction_file)
        # Log every epoch evaluation to wandb
        wandb_logging_norw(epoch, wandb, optimizer, loss, score_dict_list=both_errant, between_epochs=True)

    # saving final evaluation results, both ERRANT and custom
    final_errant_eval = both_errant # list with dicts
    final_predictions = prediction_file # file path
    final_custom_eval = custom_dict # dict
    print(f"Predictions written to file: {final_predictions}")

    sec_duration = time.time() - start 
    duration = str(timedelta(seconds = sec_duration))

    # create a file with info on this run: cmd-arguments and results
    create_runfile_norw(args, final_errant_eval, device, intermediate_results, duration, final_custom_eval)
    print("Runfile created at: ", args.description, ".txt")

    # save the checkpoint
    print("Now saving model")
    PATH = args.description + ".pt"
    torch.save({
            'epoch': epoch+1,
            'model_state_dict': my_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            "args": args,
            }, PATH)
    print(f"Model saved succesfully as {args.description}.pt !")


if __name__ == '__main__':
    main()
