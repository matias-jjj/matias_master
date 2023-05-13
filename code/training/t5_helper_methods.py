"""
Some additional methods used when training with t5_train.py
10.03.23
16.03.23
"""
#import
import torch
import wandb # in use??
from datetime import datetime
import pandas

# discard these imports?
import transformers
import numpy as np
from torch import nn
from argparse import ArgumentParser

def set_device():
    """
    Decide which device to use for tensors: CPU / GPU (cuda)
    """
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("\nCuda detected / GPU in use now")
    else:
        print("\nCuda NOT detected / CPU in use now")
    return device

def wandb_logging(epoch_number, wandb_object, optimizer, loss, score_dict, between_epochs=False):
    if between_epochs: # between each epoch
        wandb_object.log({
                    "epoch": epoch_number+1,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "true positives": score_dict["tp"],
                    "false positives": score_dict["fp"],
                    "false negatives": score_dict["fn"],
                    "precision": score_dict["precision"],
                    "recall": score_dict["recall"],
                    "f_half": score_dict["f_half"],
                })
    if not between_epochs: # between each step (batch)
        wandb_object.log({
                "epoch": epoch_number+1,
                # "grad_norm": grad_norm.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train/loss": loss.item()
            })
        
def wandb_logging_norw(epoch_number, wandb_object, optimizer, loss, score_dict_list, between_epochs=False):
    # different because 2 evaluation sets: we have 2 dicts
    if between_epochs: # between each epoch
        mapping = ["raw", "exp"]
        for i, score_dict in enumerate(score_dict_list):
            wandb_object.log({
                        f"{mapping[i]}-epoch": epoch_number+1,
                        f"{mapping[i]}-learning_rate": optimizer.param_groups[0]['lr'],
                        f"{mapping[i]}-true positives": score_dict["tp"],
                        f"{mapping[i]}-false positives": score_dict["fp"],
                        f"{mapping[i]}-false negatives": score_dict["fn"],
                        f"{mapping[i]}-precision": score_dict["precision"],
                        f"{mapping[i]}-recall": score_dict["recall"],
                        f"{mapping[i]}-f0.5": score_dict["f_half"],
                    })
    if not between_epochs: # between each step (batch), nothing special for NORW
        wandb_object.log({
                "epoch": epoch_number+1,
                # "grad_norm": grad_norm.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train/loss": loss.item()
            })

def merge_datasets(base_df, extra_dataset_path):
    """
    Adding more training data
    Please include .tsv / .txt / .csv
    """
    if extra_dataset_path[-3:] not in ["tsv", "csv"]:
        print("The file-ending of extra dataset needs a tsv/csv ending")
        return False
    xtra_dataset_df = pandas.read_csv(extra_dataset_path, sep='\t', header=0) # removes source/target on top
    new_df = pandas.concat([base_df, xtra_dataset_df], ignore_index=True) # IgIn because we dont want old indices
    return new_df

def create_runfile(args, score_dict, device, intermediate_results, duration):
    """
    Create text-file summarising the current run h-params, results, and development during training
    Try to add date here!
    """

    filename = args.description + ".txt"
    runfile = open(filename, "w")
    runfile.write("Device used for this run: " + str(device) + "\n")
    runfile.write("Duration: " + str(duration) + "\n")
    for arg in vars(args):
        runfile.write(str(arg) + "------->" + str(getattr(args, arg)) + "\n")
    for score_name in score_dict:
        runfile.write(str(score_name) + "----->" + str(score_dict[score_name]) + "\n")
    runfile.write("\nIntermediate results:\n")
    for epoch, f_half in intermediate_results:
        runfile.write("Epoch number: " + str(epoch) + "----->" + str(f_half) + "\n")
    runfile.close()

def create_runfile_norw(args, errant_eval_list, device, intermediate_results, duration, custom_eval):
    """
    Create text-file summarising the current run h-params, results, and development during training
    Try to add date here!
    Special for Norwegian: score_dict_(list) consists of 2 dicts: one for each eval-set
    Special for Norwegian: additional ask-tag-based evaluation dict
    """

    filename = args.description + ".txt"
    runfile = open(filename, "w")
    runfile.write("Device used for this run: " + str(device) + "\n")
    runfile.write("Duration: " + str(duration) + "\n")
    # run arguments
    for arg in vars(args):
        runfile.write(str(arg) + "------->" + str(getattr(args, arg)) + "\n")
    # custom evaluation
    for score_name in custom_eval:
        runfile.write("Custom EXP evaluation:" + str(score_name) + "----->" + str(custom_eval[score_name]) + "\n")
    # errant evaluation
    for score_name in errant_eval_list[0]:
        runfile.write("RAW evaluation:" + str(score_name) + "----->" + str(errant_eval_list[0][score_name]) + "\n")
    for score_name in errant_eval_list[1]:
        runfile.write("EXP evaluation:" + str(score_name) + "----->" + str(errant_eval_list[1][score_name]) + "\n")
    # old manual strategy
    runfile.write("\nIntermediate results:\n")
    for epoch, f_half in intermediate_results:
        runfile.write("Epoch number: " + str(epoch) + "----->" + str(f_half) + "\n")
    # close
    runfile.close()