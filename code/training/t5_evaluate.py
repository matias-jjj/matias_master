# module to be called by t5_train.py
# 14.03.23 - 22.03.23 - 07-04.23

#imports

# confirmed
import os # paths
import tqdm
import subprocess
import time
# not-confirmed
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW, Adam
import numpy as np
import transformers

def change_pads(original_encodings, device):
    """
    Collator converted pad-id 0 to -100 because PyTorch-CrEnLo
    this method changes them back because Huggingface tokenizer decoding
    """
    modified_encodings = []
    for encoding in original_encodings.input_ids:
        modified_encoding = [token if token != -100 else 0 for token in encoding]
        modified_encodings.append(modified_encoding)
    original_encodings.input_ids = torch.tensor(modified_encodings).to(device)
    return original_encodings

def write_temp_file(sent_list, common_path, out_filename):
    """
    create text-file from list
    called for source, target and preds separately
    """
    full_path = common_path + out_filename
    with open(full_path, 'wt') as out_file: 
        for i in range(len(sent_list)):
            out_file.write(f"{sent_list[i]}\n")
    out_file.close()
    return full_path

def write_inspection_file(args, originals, golds, preds, out_filename):
    """
    create text-file from multiple list: original /t golds /t preds
    tab-separated
    only used for manual inspection of predictions
    created after each epoch, but only the last one is saved
    """
    full_path = args.description + "_" + out_filename # not saved in subfolder
    with open(full_path, 'wt') as out_file: 
        for i in range(len(originals)):
            out_file.write(f"{originals[i]}\t{golds[i]}\t{preds[i]}\n")
    out_file.close()
    # BONUS: extra file for short sents where source != target
    bonus_file_path = args.description + "_easy_inspection.txt"
    with open(bonus_file_path, 'wt') as out_file: 
        for i in range(len(originals)):
            if originals[i] != golds[i] and len(originals[i]) < 50:
                out_file.write(f"{originals[i]}\n{golds[i]}\n{preds[i]}\n\n")
    out_file.close()

    return full_path

def generate_m2(filename_1, filename_2, common_path, out_filename, print_text):
    """
    use 2 txt-files to generate 1 m2-file
    """
    arg1 = "errant_parallel"
    arg2 = "-orig " +  filename_1 
    arg3 = "-cor " +  filename_2
    arg4 = "-out " + common_path + out_filename
    errant_commands = rf"{arg1} {arg2} {arg3} {arg4}" # note that spaces are added here
    result = subprocess.run(errant_commands, shell=True, capture_output=True, text=True) # returns CompletedProcess-object
    out_filepath = common_path + out_filename
    return out_filepath

def extract_scores_simple(errant_output):
    """
    correction-span based scores
    ERRANT output is a clumsy string ment for CMD-reading
    returns a dict instead, to be read in train-script
    """
    scores = errant_output.stdout.split("F0.5")[1]
    scores = scores.split("\t")
    tp = float(scores[0].split("\n")[1])
    fp = float(scores[1])
    fn = float(scores[2])
    precision = float(scores[3])
    recall = float(scores[4])
    f_half = float(scores[5].split("\n")[0])
    scores = {"tp":tp, "fp":fp, "fn":fn, "precision":precision, "recall":recall, "f_half":f_half }
    return scores

def extract_scores_complex(errant_output):
    """
    IKKE TILPASSET ENNÅ!!!

    correction-span based scores
    ERRANT output is a clumsy string ment for CMD-reading
    returns a dict instead, to be read in train-script
    """
    scores = errant_output.stdout.split("F0.5")[1]
    scores = scores.split("\t")
    tp = float(scores[0].split("\n")[1])
    fp = float(scores[1])
    fn = float(scores[2])
    precision = float(scores[3])
    recall = float(scores[4])
    f_half = float(scores[5].split("\n")[0])
    scores = {"tp":tp, "fp":fp, "fn":fn, "precision":precision, "recall":recall, "f_half":f_half }
    return scores

def granular_evaluation(arg1,arg2,arg3,description):
    """
    IF in testing phase (not dev)
    After final epoch, create file with more granular evaluations on error-types
    There are three different options, we use them all for further analysis
    """
    gran_file_name = description + "gran.txt"
    gran_file = open(gran_file_name, "w")
    gran_file.write(f"Granular evaluation for {description}\n")
    for level in [1,2,3]: # three levels that are arguments to ERRANT -cat
        gran_file.write(f"\nGranularity level {level}\n")
        errant_commands = rf"{arg1} {arg2} {arg3} -cat {level}"
        result = subprocess.run(errant_commands, shell=True, capture_output=True, text=True)
        gran_file.write(result.stdout)
    gran_file.close()
    return

def evaluate_errant(args, model, optimizer, tokenizer, eval_loader, device, final_epoch):
    """
    This is the main method, which is called from t5_train.py
    All the others are helper/local methods
    """

    # folder where files needed by ERRANT are stored temporarely
    common_path = os.getcwd() + "/errant_temp/" + args.description + "/" # returns string
    os.makedirs(common_path[:-1], exist_ok=True) # dir needs to exist to initiate files on the full path

    print("Evaluating...")
    # loop with DataLoader, extract all sents, and save to lists
    orig_list, gold_list, pred_list = [], [], []
    eval_iterator = tqdm.tqdm(eval_loader)
    for source, target in eval_iterator: 
        optimizer.zero_grad(set_to_none=True)
        # generate predictions
        generated_ids = model.generate(
            input_ids = source.input_ids, 
            attention_mask = source.attention_mask,
            do_sample = False, # dette forstår jeg ikke ennå 
            max_length = args.generation_max_length, 
            top_k = 0, 
            temperature = args.temperature # 1.0 is default
                ).to(device)
        # decode all ids
        target = change_pads(target, device) # convert back from -100 --> 0
        originals = tokenizer.batch_decode(source.input_ids.squeeze(), skip_special_tokens=True) # returns list
        golds = tokenizer.batch_decode(target.input_ids.squeeze(), skip_special_tokens=True)
        predictions = tokenizer.batch_decode(generated_ids.squeeze(), skip_special_tokens=True)
        # add all to lists (this can be merged with above later 15.03)
        orig_list += originals
        gold_list += golds
        pred_list += predictions

    # write to separate files
    print("Writing from lists to file")
    originals_path = write_temp_file(orig_list, common_path, "originals.txt")
    golds_path = write_temp_file(gold_list, common_path, "golds.txt")
    preds_path = write_temp_file(pred_list, common_path, "preds.txt")
    # write to common file, for inspection in the end. Returned as is
    inspection_path = write_inspection_file(args, orig_list, gold_list, pred_list, args.pred_file)

    # use ERRANT/subprocess sep-files --> m2 --> compare
    print("using ERRANT to calculate scores")

    # generate m2 for gold standard
    print_text = "\nM2 for original + gold generated: " # not in use inside m2 22.03.23
    filename_m2_golds = generate_m2(originals_path, golds_path, common_path, "gold_m2.txt", print_text)

    # generate m2 for predictions
    print_text = "\nM2 for original + pred generated: " # not in use inside m2 22.03.23
    filename_m2_preds = generate_m2(originals_path, preds_path, common_path, "pred_m2.txt", print_text)

    # compare the two M2-files and extract scores (no file returned)
    arg1 = "errant_compare"
    arg2 = "-hyp " + filename_m2_preds
    arg3 = "-ref " + filename_m2_golds
    errant_commands = rf"{arg1} {arg2} {arg3}"
    result_basic = subprocess.run(errant_commands, shell=True, capture_output=True, text=True)
    if args.testing and final_epoch: # more granular evaluation, create file
        print("Now running granular evaluation/creating file\n")
        granular_evaluation(arg1, arg2, arg3, args.description)
    scores = extract_scores_simple(result_basic)
    # return basic results to main-script
    return scores, inspection_path

    