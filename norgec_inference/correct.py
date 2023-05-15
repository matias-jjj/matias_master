try:
    from argparse import ArgumentParser
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, logging
    from modeling_nort5 import NorT5ForConditionalGeneration
    import time
    from prettytable import PrettyTable as pt
except:
    print("Du må aktivere enivronment før du kan bruke dette programmet!")
    print("Skriv: conda activate seq2seq_baseline_2 i terminalen")
    exit()


"""
CUSTOM MANUAL VERSION
15.12.23
Matias Jentoft master thesis
Grammatical Error Correction
Inference for the models in the norgec-class
"""
#logging.set_verbosity_warning()
logging.set_verbosity_error()

# choose model
print("Velkommen til NORGEC, her kan du få rettet setningene dine")
print("pssst...NORGEC står for Norwegian Grammatical Error Correction")
valid_choice = False
while not valid_choice:
    selected_model = input("Hvilke(n) modell(er) vil du bruke: nort5, byt5, mt5 eller alle? ")
    if selected_model == "alle":
        print("\nDu kan ikke velge alle dessverre, maskinen min tåler det ikke!\n")
    elif selected_model in ["nort5", "byt5", "mt5"]:
        valid_choice = True
        print("Du har valgt følgende modell for gec:", selected_model)
    else:
        print("Du valgte ", selected_model, ", det var ikke et gyldig valg.")

all_models = [] # one or three models

if selected_model in ["nort5", "alle"]:
    print("Laster inn norgec_nort5...")
    model_address = "MatiasJ/norgec_nort5"
    tokenizer = AutoTokenizer.from_pretrained(model_address, model_max_length=512)
    model_itself = NorT5ForConditionalGeneration.from_pretrained(model_address)
    gen_max_length = 50
    model = {"name":"nort5", "mod":model_itself, "tok":tokenizer, "gml":gen_max_length}
    all_models.append(model)
if selected_model in ["byt5", "alle"]:
    print("Laster inn norgec_byt5...")
    model_address = "MatiasJ/norgec_byt5"
    tokenizer = AutoTokenizer.from_pretrained(model_address, model_max_length=512)
    model_itself = AutoModelForSeq2SeqLM.from_pretrained(model_address)
    gen_max_length = 250
    model = {"name":"nort5", "mod":model_itself, "tok":tokenizer, "gml":gen_max_length}
    all_models.append(model)
if selected_model in ["mt5", "alle"]:
    print("Laster inn norgec_mt5")
    model_address = "MatiasJ/norgec_mt5"
    tokenizer = AutoTokenizer.from_pretrained(model_address, model_max_length=512)
    model_itself = AutoModelForSeq2SeqLM.from_pretrained(model_address)
    gen_max_length = 50
    model = {"name":"byt5", "mod":model_itself, "tok":tokenizer, "gml":gen_max_length}
    all_models.append(model)

# start collecting input from the user and generate
user_input = "hei"
while user_input != "avslutt":
    user_input = input("\nSkriv inn setningen du vil ha rettet(*avslutt* for å avslutte): ")
    if user_input == "avslutt":
        break
    predictions = []
    for model in all_models:
        source_tokenized = model["tok"](user_input, return_tensors='pt')
        output = model["mod"].generate(
            input_ids=source_tokenized.input_ids, 
            attention_mask=source_tokenized.attention_mask, 
            do_sample=True, 
            max_length=model["gml"]
            )
        prediction = tokenizer.decode(output.squeeze(), skip_special_tokens=True)
        predictions.append((model["name"], prediction))

    # show results
    tb = pt()
    #Add rows
    tb.add_row(["Opprinnelig setning",user_input])
    for prediction in predictions:
        tb.add_row([prediction[0],prediction[1]])
    tb.header = False
    print(tb)
    # print("\nOpprinnelig: ", user_input)
    # for prediction in predictions:
    #     print(prediction[0], "       ", prediction[1])
    time.sleep(2)

# avslutter
print("\nAvslutter. Takk for at du ville bruke NORGEC, håper opplevelsen var fornøyelig")
time.sleep(2)