from importlib import import_module
import spacy
from errant.annotator import Annotator
import time
import sys
sys.path.append('../errant')


# ERRANT version
__version__ = '2.3.3'

# Load an ERRANT Annotator object for a given language
def load(lang, nlp=None):
    # start debugging by interchanging these two names
    lang_1 = "en"
    lang_2 = "en_core_web_sm"
    # Make sure the language is supported
    supported = {"en", "en_core_web_sm"}
    if lang_1 not in supported:
        raise Exception("%s is an unsupported or unknown language" % lang_1)
    # Load spacy
    nlp = nlp or spacy.load(lang_2, disable=["ner"]) # works

    # Load language edit merger
    merger = import_module("errant.%s.merger" % lang_1) # Problem

    # Load language edit classifier
    classifier = import_module("errant.%s.classifier" % lang_1)
    # The English classifier needs spacy
    if lang_1 == "en": classifier.nlp = nlp

    # Return a configured ERRANT annotator
    return Annotator(lang_1, nlp, merger, classifier)
