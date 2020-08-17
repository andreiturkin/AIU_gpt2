import pandas as pd
import pickle
from fastai.basics import get_files
import os

def parser(folder):
    files = get_files(folder, '.xlsx')
    resData = []
    for f in files:
        df = pd.ExcelFile(f).parse(header=None)
        df.dropna(inplace=True)

        # check the number of columns
        if len(df.columns) > 3:
            print('Too many columns to parse. Skipped: {}'.format(f))
            continue

        if len(df.columns) == 3:
            df.drop(columns=[0], inplace=True)
            df.columns = [0, 1]

        if 1 in df.columns:
            d = dict(zip([s.rstrip() for s in df[0].to_list()], df[1].to_list()))
            if 'Абстракт' not in d:
                print('Empty abstract. Skipped: {}'.format(f))
                continue
            resData.append(dict(zip(['Title', 'Abstract'], [d['Название'], d['Абстракт']])))
        else: print('Empty file. Skipped: {}'.format(f))

    return resData

def save2pkl(f, d):
    with open(f, 'wb') as f:
        pickle.dump(d, f)

def openpkl(f):
    with open(f, 'rb') as f:
        obj = pickle.load(f)
    return obj

if __name__ == '__main__':
    folders2parse = ['./data/Мед статьи 1', './data/Мед статьи 2']
    file2save = './data/dataset.pkl'
    Data = []
    if os.path.isfile(file2save):
        Data = openpkl(file2save)
    else:
        for folder in folders2parse:
            Data.extend(parser(folder))

        print('Parsing is done with {} title-abstract pairs'.format(len(Data)))
        print('Saving data to a .pkl file: {}'.format(file2save))
        save2pkl(file2save, Data)


    # partially adopted from corpus.ipynb
    #                           Apache License
    #                       Version 2.0, January 2004
    #                    http://www.apache.org/licenses/
    split = int(len(Data)*0.9)
    NEW_LINE = '<|n|>'

    # training data
    for i, d in enumerate(Data[:split]):
        with open('./data/trainData/t_abstracts_{}.txt'.format(i), 'w') as c:
            c.write(d['Abstract'].replace('\n', f' {NEW_LINE} ') + '\n')

    # validation data
    for i, d in enumerate(Data[split:]):
        with open('./data/evaluateData/v_abstracts_{}.txt'.format(i), 'w') as c:
            c.write(d['Abstract'].replace('\n', f' {NEW_LINE} ') + '\n')

    # tokenization
    from fastai.basics import *
    from multiprocessing import Pool
    import regex as re
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException
    import os,sys,inspect
    from run_lm_finetuning import TextDataset
    from yt_encoder import YTEncoder

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

    txts = get_files('./data/trainData', '.txt')


    def cache_fn(fn):
        tokenizer = YTEncoder.from_pretrained('./bpe/yt.model')
        TextDataset.process_file(fn, tokenizer, 1024, shuffle=True)

    for _ in progress_bar(Pool(32).imap_unordered(cache_fn, txts), len(txts)):
        pass




