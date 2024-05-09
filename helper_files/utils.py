'''
Running utils.py with download and format datasets to train, val, test splits,
Corrects for some inconsistent files (color)
Saves to local folder

Additional utils functions:
remove_classes: Remove unwanted class types

'''

from datasets import load_dataset
from PIL import Image
import numpy as np

def transform(dset):
    dset['image'] = [pil_img.convert(mode='RGB') for pil_img in dset['image']]
    return dset

def save_dataset_to_local(save_path ,config='160px', train_val_split=0.15):
    # load imagenette from huggingface
    # config options are '160px', '320px', or 'full_size' resolutions
    datasetDict = load_dataset("frgfm/imagenette", name=config)

    # dataset only includes train and validation.
    # set test = validation
    # split train into a train and validation split
    # rename "validation" dataset -> "test"
    test = datasetDict.pop("validation")
    datasetDict["test"] = test

    # create train validation split
    train_original = datasetDict['train']
    train, val= train_original.train_test_split(test_size=train_val_split, stratify_by_column='label').values()

    # store new values to datasetDict
    datasetDict['train'] = train
    datasetDict['validation'] = val

    # load test dataset
    test = datasetDict['test']

    # ensure entire dataset is rgb
    for ds in datasetDict:
        datasetDict[ds] = datasetDict[ds].map(transform, batched=True, batch_size=250, writer_batch_size=250)

    # save dataset
    datasetDict.save_to_disk(save_path)
    pass

def remove_classes(datasetDict, classes_to_remove):
    datasetDict = datasetDict.filter(lambda x: x['label'] not in classes_to_remove)
    class_nums = datasetDict['train'].unique('label')
    class_nums.sort()
    class_names = datasetDict['train'].features['label'].int2str(class_nums)
    labels = dict(zip(class_nums, class_names))

    return datasetDict, labels


if __name__ == '__main__':
    configs = ['160px', '320px']
    save_path = ['datasets/local_dataset_'+config for config in configs]
    for i, config in enumerate(configs):
        save_dataset_to_local(save_path=save_path[i], config=config, train_val_split=0.15)

