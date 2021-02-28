import util
import json
import os
import time
from google_trans_new import google_translator

ood_train_dir = 'datasets/oodomain_train/'
ood_train_aug_dir = 'datasets/oodomain_train_aug/'
ood_val_dir = 'datasets/oodomain_val/'
ood_val_aug_dir = 'datasets/oodomain_val_aug/'

# def backTranslate(translator, sentence, dest):
#     print(sentence)
#     forward = translator.translate(sentence, lang_src='en', lang_tgt=dest)
#     backward = translator.translate(forward, lang_src=dest, lang_tgt='en')
#     return backward
#
# def main():
#     translator = google_translator()
#     for dataset in os.listdir(ood_train_dir):
#         dataset_dict_curr = util.read_squad(f'{ood_train_dir}/{dataset}')
#         for i in range(len(dataset_dict_curr['question'])):
#             dataset_dict_curr['question'][i] = backTranslate(translator, dataset_dict_curr['question'][i], 'es')
#             dataset_dict_curr['context'][i] = backTranslate(translator, dataset_dict_curr['context'][i], 'es')

# def stringify(lst, delim):
#     string = ""
#     for elem in lst:
#         string += elem + delim
#     return string
#
# def augment_by_chunk(translator, lst):
#     delim = '\n'
#     print(f"Length of initial list is: {len(lst)}")
#     # for i in range(10):
#     #     print(lst[i])
#     # input()
#     step = 2000
#     string = stringify(lst, delim)
#     print(f"Length of initial string is: {len(string)}")
#     aug = ""
#     chunks = [string[i:i+step] for i in range(0, len(string), step)]
#     for chunk in chunks:
#         # print(chunk)
#         # print(chunk.count(delim))
#         back = backTranslate(translator, chunk, 'es')
#         # print(back)
#         # print(back.count(delim))
#         aug += back
#         # input()
#     print(f"Length of augmented string is {len(aug)}")
#     return_list = aug.strip('][').split(delim)
#     print(f"Length of augmented list is: {len(return_list)}")
#     # for i in range(10):
#     #     print(return_list[i])
#     input()
#     return return_list

# def augment_data(dataset_dict_curr):
#     translator = google_translator()
#     for i in range(len(dataset_dict_curr['question'])):
#
#     dataset_dict_curr['question'] = augment_by_chunk(translator, dataset_dict_curr['question'])
#     dataset_dict_curr['context'] = augment_by_chunk(translator, dataset_dict_curr['context'])
#     return dataset_dict_curr

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
    for i in range(len(dataset_dict_curr['question'])):
        dataset_dict_curr['question'][i] = backTranslate(translator, dataset_dict_curr['question'][i], 'es')
        dataset_dict_curr['context'][i] = backTranslate(translator, dataset_dict_curr['context'][i], 'es')
    # dataset_dict_curr['question'] = augment_by_chunk(translator, dataset_dict_curr['question'])
    # dataset_dict_curr['context'] = augment_by_chunk(translator, dataset_dict_curr['context'])

def main():
    translator = google_translator()
    for dataset in os.listdir(ood_val_dir):
    #     with open(ood_train_dir[:-1] + '_aug/' + dataset + '.json', 'rb') as f:
    #         aug_dict = json.load(f)
    #         print(aug_dict['question'][0])
        print(f"Augmenting {dataset}")
        dataset_dict_curr = util.read_squad(f'{ood_val_dir}/{dataset}')
        print(dataset_dict_curr['context'][0])
        augment_data(dataset_dict_curr)
        print(dataset_dict_curr['context'][0])
        print(f"Saving {dataset}")
        with open(f"{ood_val_aug_dir}{dataset}.json", 'w') as f:
            json.dump(dataset_dict_curr, f)

if __name__ == '__main__':
    main()
