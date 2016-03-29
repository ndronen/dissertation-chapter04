import sys
import re
import os
import json
import collections
import importlib
import modeling.utils
from modeling.dataset import HDF5FileDataset
from glob import glob
import numpy as np
import pandas as pd

from spelling.baseline import CharacterLanguageModel, LanguageModelClassifier
from spelling.utils import build_progressbar as build_pbar

from sklearn.metrics import accuracy_score

def load_config(model_dir):
    return json.load(open(model_dir + '/config.json'))

def load_model_module(model_dir):
    sys.path.append('.')
    model_module_path = model_dir.replace('/', '.') + '.model'
    return importlib.import_module(model_module_path)

def load_model(model_dir, model_weights='model.h5'):
    config = load_config(model_dir)
    config['model_weights'] = model_dir + '/' + model_weights
    model_cfg = modeling.utils.ModelConfig(**config)

    train_data = HDF5FileDataset(
        os.path.join(model_cfg.data_dir, model_cfg.train_path),
        model_cfg.data_name,
        [model_cfg.target_name],
        model_cfg.batch_size,
        model_cfg.seed)

    M = load_model_module(model_dir)

    if model_cfg.n_residual_blocks > 0:
        model = M.build_graph(model_cfg, train_data)
    else:
        model = M.build_sequential(model_cfg, train_data)

    return model, model_cfg, train_data

def train_lm(pos_words, neg_words, discount='witten-bell', order=3, debug=False):
    lm_pos = CharacterLanguageModel(discount, order, debug=debug)
    lm_pos.fit(pos_words)
    lm_neg = CharacterLanguageModel(discount, order, debug=debug)
    lm_neg.fit(neg_words)
    return LanguageModelClassifier([lm_pos, lm_neg])

def mark_word(word):
    return '^' + word + '$'

def build_index(char_matrix, words):
    marked_words = [mark_word(w) for w in words]
    char_to_index = {}
    max_index = np.max(char_matrix)
    i = 0
    while len(char_to_index) < max_index:
        row = char_matrix[i]
        word = marked_words[i]
        for j,val in enumerate(row):
            try:
                ch = word[j]
                char_to_index[ch] = val
            except IndexError:
                continue
        i += 1
    return char_to_index

def run(model_dir='models/artificial_errors/d1e4e4c6f36311e5aa8efcaa149e39ea', **kwargs):
    # Load ConvNet.
    model, model_cfg, train_data = load_model(model_dir)

    csv_name = os.path.splitext(model_cfg.validation_path)[0] + '.csv'
    csv_path = model_cfg.data_dir + '/' + csv_name
    df = pd.read_csv(csv_path, sep='\t', encoding='utf8')
    # There are three words that are nan in this data set.
    df.loc[df.word.isnull(), 'word'] = 'UNK'
    pos_words = df[df.binary_target == 1].word.tolist()
    neg_words = df[df.binary_target == 0].word.tolist()

    # Build index 
    char_to_index = build_index(
            train_data[model_cfg.data_name[0]], 
            df.word.tolist())
    input_width = train_data[model_cfg.data_name[0]].shape[1]

    # Load language model.
    lm = train_lm(pos_words, neg_words, **kwargs)

    # Get model probabilities from the ConvNet and the character language
    # model for several data sets:
    # - Aspell English dictionary and synthetic non-words.
    # - Corpora of company or brand names.
    # - Corpus of North American cities.

    def predict_proba(words):
        examples = np.zeros((len(df), input_width))
        ok = []
        ok_words = []

        for i,word in enumerate(words):
            try:
                for j,ch in enumerate(mark_word(word)):
                    examples[i,j] = char_to_index[ch]
            except (IndexError, KeyError):
                # Skip words that are too long or that contain a character
                # not in our index.
                continue

            ok.append(i)
            ok_words.append(word)

        examples = examples[ok]
        model_proba = model.predict_proba(examples)
        lm_proba = lm.predict_proba(ok_words)

        return model_proba, lm_proba, ok_words

    def predict_proba_corpus(corpus_path=None, words=None, encoding='utf8'):
        if words is None:
            with open(corpus_path, 'r', encoding=encoding) as f:
                words = [w.strip() for w in f.read().split('\n')]
        words = [w for w in words if isinstance(w, str)]
        lengths = [len(w) for w in words]
        print(len(words), min(lengths), max(lengths))
        corpus_probas = predict_proba(words)
        probas = {}
        probas['ConvNet'] = corpus_probas[0]
        probas['LM'] = corpus_probas[1]
        probas['word'] = corpus_probas[2]
        return probas

    aspell_probas = predict_proba(df.word.tolist())

    probas = collections.defaultdict(dict)

    probas['Aspell']['ConvNet'] = aspell_probas[0]
    probas['Aspell']['LM'] = aspell_probas[1]
    probas['Aspell']['word'] = df.word.tolist()
    probas['Aspell']['target'] = df.binary_target.tolist()

    corpora = glob("data/brand*.txt")
    corpora.append("data/north-american-cities.txt")

    for corpus_path in corpora:
        corpus_name = corpus_path.replace('data/', '').replace('.txt', '')
        try:
            corpus_probas = predict_proba_corpus(corpus_path)
        except UnicodeDecodeError:
            corpus_probas = predict_proba_corpus(corpus_path,
                    encoding='latin1')
        probas[corpus_name] = corpus_probas

    english_csv = os.environ['HOME'] + "/proj/spelling/data/aspell-dict.csv.gz"
    english_df = pd.read_csv(english_csv, sep='\t', encoding='utf8')
    english_vocab = set(english_df.word.tolist())
    dict_csvs = glob(os.environ['HOME'] + "/proj/spelling/data/aspell-dict*.csv")

    for dict_csv in dict_csvs:
        dict_df = pd.read_csv(dict_csv, sep='\t', encoding='utf8')
        dict_vocab = dict_df.word.tolist()
        non_english_vocab = set(dict_vocab).difference(english_vocab)
        dict_probas = predict_proba_corpus(words=non_english_vocab)

        dict_name = re.sub('.*aspell-dict-', '', dict_csv)
        dict_name = dict_name.replace('.csv', '')

        probas[dict_name] =  dict_probas

    return convert(probas)

def convert(probas):
    name = []
    p0 = []
    p1 = []
    model = []
    word = []
    target = []
    for dataset in probas.keys():
        print(dataset)
        for model_name in ['ConvNet', 'LM']:
            print(model_name)
            probs = probas[dataset][model_name]
            p0.extend(probs[:, 0])
            p1.extend(probs[:, 1])
            name.extend([dataset] * len(probs))
            model.extend([model_name] * len(probs))
            word.extend(probas[dataset]['word'])
            if 'target' in probas[dataset]:
                target.extend(probas[dataset]['target'])
            else:
                target.extend([None] * len(probs))

    # Normalize the language model model probabilities.  The ConvNet
    # model's probabilities are already normalized; for them this
    # is no-op.
    p0 = np.array(p0)
    p1 = np.array(p1)
    p0norm = p0/(p0 + p1)
    p1norm = p1/(p0 + p1)
    p0 = p0norm
    p1 = p1norm

    return pd.DataFrame(data={
        'model': model,
        'word': word,
        'target': target,
        'p0': p0,
        'p1': p1,
        'dataset': name
        })

def build_accuracy_data_frame(df):
    datasets = df.dataset.unique()
    models = df.model.unique()

    results = []

    for ds in datasets:
        for model in models:
            df_tmp = df[(df.dataset == ds) & (df.model == model)]
            targets = [0] * len(df_tmp)
            accuracy = accuracy_score(
                    targets,
                    (df_tmp.p1 > 0.5).astype(int))
            results.append({
                'dataset': ds,
                'model': model,
                'Accuracy': accuracy
                })


    results_df = pd.DataFrame(data=results)

    count_map = df.dataset.value_counts().to_dict()
    results_df['N'] = results_df.dataset.apply(
            lambda d: count_map[d])

    dataset_map = {
            'br': 'Breton',
            'ca': 'Catalan',
            'cs': 'Czech',
            'cy': 'Welsh',
            'de': 'German',
            'es': 'Spanish',
            'et': 'Estonian',
            'fr': 'French',
            'ga': 'Irish (Gaeilge)',
            'hsb': 'Upper Sorbian',
            'is': 'Icelandic',
            'it': 'Italian',
            'nl': 'Dutch',
            'sv': 'Swedish'
            }
            
    # Only include datasets in the map.
    results_df = results_df[results_df.dataset.isin(dataset_map.keys())]

    # And rename them for presentation.
    results_df.dataset = results_df.dataset.apply(
            lambda s: dataset_map[s] if s in dataset_map else s)

    # Upper case the column names.
    results_df.columns = [c.title() for c in results_df.columns]

    results_df['Language'] = results_df.Dataset
    del results_df['Dataset']

    results_df = results_df.sort_values(['Language', 'Model'])

    return results_df[['Dataset', 'N', 'Model', 'Accuracy']]
