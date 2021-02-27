import util
import json
import os
from google_trans_new import google_translator

ood_train_dir = 'datasets/oodomain_train/'
ood_train_aug_dir = 'datasets/oodomain_train_aug/'

def backTranslate(translator, sentence, dest):
    print(sentence)
    forward = translator.translate(sentence, lang_src='en', lang_tgt=dest)
    backward = translator.translate(forward, lang_src=dest, lang_tgt='en')
    return backward

def main():
    translator = google_translator()
    for dataset in os.listdir(ood_train_dir):
        dataset_dict_curr = util.read_squad(f'{ood_train_dir}/{dataset}')
        for i in range(len(dataset_dict_curr['question'])):
            dataset_dict_curr['question'][i] = backTranslate(translator, dataset_dict_curr['question'][i], 'es')
            dataset_dict_curr['context'][i] = backTranslate(translator, dataset_dict_curr['context'][i], 'es')

if __name__ == '__main__':
    main()
