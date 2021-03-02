import util
import json
import os
import time
from google_trans_new import google_translator

ood_train_dir = 'datasets/oodomain_train/'
ood_train_aug_dir = 'datasets/oodomain_train_aug/'
ood_val_dir = 'datasets/oodomain_val/'
ood_val_aug_dir = 'datasets/oodomain_val_aug/'

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

def augment_data(dataset_dict_curr):
    translator = google_translator()
    max_context_len = 0
    max_question_len = 0
    for i in range(len(dataset_dict_curr['question'])):
        question = backTranslate(translator, dataset_dict_curr['question'][i], 'es')
        context = backTranslate(translator, dataset_dict_curr['context'][i], 'es')
        max_context_len = max(max_context_len, len(context))
        max_question_len = max(max_question_len, len(question))
        dataset_dict_curr['question'][i] = backTranslate(translator, dataset_dict_curr['question'][i], 'es')
        dataset_dict_curr['context'][i] = backTranslate(translator, dataset_dict_curr['context'][i], 'es')
        dataset_dict_curr['id'][i] += 'a'
    for j in range(len(dataset_dict_curr['question'])):
        dataset_dict_curr['question'][j] += PAD_CHAR * (max_question_len - len(dataset_dict_curr['question'][j]))
    for k in range(len(dataset_dict_curr['context'])):
        dataset_dict_curr['context'][k] += PAD_CHAR * (max_context_len - len(dataset_dict_curr['context'][k]))


def main():
    dir = ood_val_dir
    save_dir = ood_val_aug_dir
    translator = google_translator()
    for dataset in os.listdir(dir):
        print(f"Augmenting {dataset}")
        dataset_dict_curr = util.read_squad(f'{dir}/{dataset}')
        augment_data(dataset_dict_curr)
        print(f"Saving {dataset}")
        with open(f"{save_dir}{dataset}.json", 'w') as f:
            json.dump(dataset_dict_curr, f)

if __name__ == '__main__':
    main()
