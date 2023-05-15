from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, pipeline

"""
AUTMATED VERSION using "pipeline" from HuggingFace
15.12.23
Matias Jentoft master thesis
Grammatical Error Correction
Inference for the models in the norgec-class
"""

# Set up arguments.
parser = ArgumentParser()
parser.add_argument("--sentence", type=str, default="Dette er setning jeg vil rette.") # important to change for each slurm!
parser.add_argument("--pretrained_model", type=str, default="byt5")

# from now on "args" represents variables that can be changed by passing from terminal
args = parser.parse_args()
print("\nYou have chosen to use the following model: ", args.pretrained_model, "\n")

source = args.sentence

if "nort5" in args.pretrained_model:
    model = "MatiasJ/norgec_nort5"
elif "byt5" in args.pretrained_model:
    model = "MatiasJ/norgec_byt5"
elif "mt5" in args.pretrained_model:
    model = "MatiasJ/norgec_mt5"

generator = pipeline(model=model)

prediction = generator(source)

print("Source: ", source)
print("Prediction: ", prediction)