import util
import json
import os
import time
import sys
import random
from google_trans_new import google_translator
from PyDictionary import PyDictionary

ood_train_dir = 'datasets/oodomain_train/'
ood_train_bt_dir = 'datasets/oodomain_train_bt/'
ood_train_syn_dir = 'datasets/oodomain_train_syn/'
alpha = 0.2

PAD_CHAR = u"\u25A1"

def backTranslate(translator, sentence, dest):
    forward = translator.translate(sentence, lang_src='en', lang_tgt=dest)
    time.sleep(1)
    backward = translator.translate(forward, lang_src=dest, lang_tgt='en')
    # print(backward)
    time.sleep(1)
    if not backward:
        print(f"Alert: {backward}")
    return backward

def augment_bt_data(translator, dataset_dict_curr):
    for i in range(len(dataset_dict_curr['question'])):
        question = backTranslate(translator, dataset_dict_curr['question'][i], 'es')
        context = backTranslate(translator, dataset_dict_curr['context'][i], 'es')
        if dataset_dict_curr['answer'][i]['answer_start'][0] < len(context):
            dataset_dict_curr['question'][i] = question
            dataset_dict_curr['context'][i] = context
            dataset_dict_curr['id'][i] += 'a'

def generateSynonyms(dictionary, dataset_dict_curr):
    for i in range(len(dataset_dict_curr['context'])):
        words = dataset_dict_curr['context'][i].split()
        k = int(float(len(words)) * alpha)
        choices = random.choices(range(len(words)), k=k)
        for choice in choices:
             synonyms = dictionary.synonym(words[choice])
             words[choice] = synonyms[0] if synonyms else words[choice]
        context = " ".join(words)
        if dataset_dict_curr['answer'][i]['answer_start'][0] < len(context):
             dataset_dict_curr['context'][i] = context
             dataset_dict_curr['id'][i] += 'b'

def main():
    backtranslate = False
    synonym = False
    if len(sys.argv) <= 1:
        input('Are you sure you want to create backtranslation AND synonym datasets?')
        backtranslate = True
        synonym = True
    elif len(sys.argv) > 2:
        print("No more that one argument please.")
        exit()
    else:
        if sys.argv[1] == 'bt':
            backtranslate = True
        elif sys.argv[1] == 'syn':
            synonym = True
    if backtranslate: translator = google_translator()
    if synonym: dictionary = PyDictionary()
    for dataset in os.listdir(ood_train_dir):
        dataset_dict_curr = util.read_squad(f'{ood_train_dir}/{dataset}')
        if backtranslate:
            copy = dataset_dict_curr.copy()
            print(f"Backtranslating {dataset}")
            augment_bt_data(translator, copy)
            print(f"Saving {dataset}")
            with open(f"{ood_train_bt_dir}{dataset}.json", 'w') as f:
                json.dump(copy, f)
        if synonym:
            if dataset != 'duorc': continue
            copy = dataset_dict_curr.copy()
            print(f"Finding synonyms for {dataset}")
            generateSynonyms(dictionary, copy)
            print(f"Saving {dataset}")
            with open(f"{ood_train_syn_dir}{dataset}.json", 'w') as f:
                json.dump(copy, f)

def playground(folder):
    for file in os.listdir(folder):
        if not file.endswith('.json'):
            continue
        new_dict = {'context': [], 'question': [], 'answer': [], 'id': []}
        aug_dict = {}
        print(f'{folder}{file}')
        with open(f'{folder}{file}') as f:
            aug_dict = json.load(f)
            print(len(aug_dict['context']))
            for i in range(len(aug_dict['context'])):
                if aug_dict['answer'][i]['answer_start'][0] < len(aug_dict['context'][i]):
                    new_dict['context'].append(aug_dict['context'][i])
                    new_dict['question'].append(aug_dict['question'][i])
                    new_dict['answer'].append(aug_dict['answer'][i])
                    new_dict['id'].append(aug_dict['id'][i])
        print(len(new_dict['context']))
        if len(new_dict['context']) > 0 and len(new_dict['context']) < len(aug_dict['context']):
            os.remove(f'{folder}{file}')
            with open(f'{folder}{file}', 'w') as f:
                new_dict = json.dump(new_dict, f)

if __name__ == '__main__':
    main()
