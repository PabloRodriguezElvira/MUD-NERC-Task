
import os

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize

from dataset import *

class Codemaps :
    GAZETTEER_JOINER = "|||"

    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None) :

        if isinstance(data,Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen)

        elif type(data) == str and maxlen is None and suflen is None:
            self.__load(data)

        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()

            
    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and 
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen) :

        self.maxlen = maxlen
        self.suflen = suflen
        self.__init_fixed_features()

        words = set()
        lc_words = set()
        sufs = set()
        labels = set()
        shapes = set()
        
        for s in data.sentences() :
            for t in s :
                words.add(t['form'])
                lc_words.add(t['lc_form'])
                sufs.add(t['lc_form'][-self.suflen:])
                labels.add(t['tag'])
                shapes.add(self.__get_shape(t['form']))

        self.word_index = {w: i+2 for i,w in enumerate(sorted(words))}
        self.word_index['PAD'] = 0 # Padding
        self.word_index['UNK'] = 1 # Unknown words

        self.lc_word_index = {w: i+2 for i,w in enumerate(sorted(lc_words))}
        self.lc_word_index['PAD'] = 0 # Padding
        self.lc_word_index['UNK'] = 1 # Unknown lowercased words

        self.suf_index = {s: i+2 for i,s in enumerate(sorted(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes

        self.shape_index = {s: i+2 for i,s in enumerate(sorted(shapes))}
        self.shape_index['PAD'] = 0  # Padding
        self.shape_index['UNK'] = 1  # Unknown token shape

        self.label_index = {t: i+1 for i,t in enumerate(sorted(labels))}
        self.label_index['PAD'] = 0 # Padding

        self.gazetteer_index = {'PAD': 0, 'NONE': 1}
        self.gazetteer_phrases = self.__load_external_gazetteer()
        gazetteer_labels = sorted(
            {
                label_key
                for label in self.gazetteer_phrases.values()
                for label_key in (f'B-{label}', f'I-{label}')
            }
        )
        for i, label in enumerate(gazetteer_labels) :
            self.gazetteer_index[label] = i + 2
        self.__rebuild_gazetteer_lookup()

    def __init_fixed_features(self) :
        self.cap_index = {
            'PAD': 0,
            'LOWER': 1,
            'TITLE': 2,
            'UPPER': 3,
            'MIXED': 4,
            'OTHER': 5,
        }
        self.num_index = {
            'PAD': 0,
            'NO': 1,
            'ALL': 2,
            'SOME': 3,
        }
        self.dash_index = {
            'PAD': 0,
            'NO': 1,
            'YES': 2,
        }

    def __shape_class(self, ch) :
        if ch.isupper() :
            return 'X'
        if ch.islower() :
            return 'x'
        if ch.isdigit() :
            return 'd'
        if ch == '-' :
            return '-'
        if ch in "/.,()" :
            return ch
        return 'o'

    def __get_shape(self, token) :
        shape = []
        last = None
        reps = 0
        for ch in token[:12] :
            shp = self.__shape_class(ch)
            if shp == last :
                reps += 1
                if reps > 2 :
                    continue
            else :
                last = shp
                reps = 1
            shape.append(shp)
        return "".join(shape) if shape else "EMPTY"

    def __get_caps_feature(self, token) :
        if token.islower() :
            return 'LOWER'
        if token.istitle() :
            return 'TITLE'
        if token.isupper() :
            return 'UPPER'
        if any(ch.isalpha() for ch in token) :
            return 'MIXED'
        return 'OTHER'

    def __get_num_feature(self, token) :
        has_digit = any(ch.isdigit() for ch in token)
        if not has_digit :
            return 'NO'
        if all(ch.isdigit() for ch in token) :
            return 'ALL'
        return 'SOME'

    def __get_dash_feature(self, token) :
        return 'YES' if '-' in token else 'NO'

    def __encode_and_pad(self, sentences, value_fn, pad_value) :
        encoded = [[value_fn(w) for w in s] for s in sentences]
        return pad_sequences(maxlen=self.maxlen, sequences=encoded, padding="post", value=pad_value)

    def __gazetteer_path(self, filename) :
        return os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "lab_resources", "DDI", "resources", filename)
        )

    def __tokenize_gazetteer_entry(self, text) :
        return tuple(token.lower() for token in word_tokenize(text.strip()) if token.strip())

    def __add_gazetteer_entry(self, entries, token_sequence, label, priority) :
        if not token_sequence :
            return
        current = entries.get(token_sequence)
        if current is None or priority > current[1] or (priority == current[1] and label < current[0]) :
            entries[token_sequence] = (label, priority)

    def __load_external_gazetteer(self) :
        entries = {}
        priorities = {'brand': 4, 'group': 3, 'drug': 2, 'hsdb': 1}

        drugbank = self.__gazetteer_path("DrugBank.txt")
        if os.path.exists(drugbank) :
            with open(drugbank, encoding="utf-8") as f :
                for line in f :
                    line = line.strip()
                    if not line or "|" not in line :
                        continue
                    entry, label = line.rsplit("|", 1)
                    label = label.strip().lower()
                    if label not in priorities :
                        continue
                    self.__add_gazetteer_entry(
                        entries,
                        self.__tokenize_gazetteer_entry(entry),
                        label,
                        priorities[label]
                    )

        hsdb = self.__gazetteer_path("HSDB.txt")
        if os.path.exists(hsdb) :
            with open(hsdb, encoding="utf-8") as f :
                for line in f :
                    entry = line.strip()
                    if not entry :
                        continue
                    self.__add_gazetteer_entry(
                        entries,
                        self.__tokenize_gazetteer_entry(entry),
                        "hsdb",
                        priorities["hsdb"]
                    )

        return {token_sequence: label for token_sequence, (label, _) in entries.items()}

    def __rebuild_gazetteer_lookup(self) :
        self.gazetteer_first_token = {}
        for token_sequence, label in self.gazetteer_phrases.items() :
            self.gazetteer_first_token.setdefault(token_sequence[0], []).append((token_sequence, label))

        for token in self.gazetteer_first_token :
            self.gazetteer_first_token[token].sort(key=lambda item: (-len(item[0]), item[1], item[0]))

    def __gazetteer_begin_label(self, label) :
        begin = f'B-{label}'
        if begin in self.gazetteer_index :
            return begin
        return label

    def __gazetteer_inside_label(self, label) :
        inside = f'I-{label}'
        if inside in self.gazetteer_index :
            return inside
        return label

    def __encode_gazetteer_sentence(self, sentence) :
        tags = ['NONE'] * len(sentence)
        lc_tokens = [token['lc_form'] for token in sentence]
        i = 0

        # Longest-match-first phrase lookup using the external lexicons.
        while i < len(lc_tokens) :
            best_match = None
            for token_sequence, label in self.gazetteer_first_token.get(lc_tokens[i], []) :
                length = len(token_sequence)
                if tuple(lc_tokens[i:i+length]) == token_sequence :
                    best_match = (length, label)
                    break

            if best_match is None :
                i += 1
                continue

            length, label = best_match
            tags[i] = self.__gazetteer_begin_label(label)
            for j in range(1, length) :
                tags[i+j] = self.__gazetteer_inside_label(label)
            i += length

        return [self.gazetteer_index[tag] for tag in tags]

    ## --------- load indexs ----------- 
    def __load(self, name) : 
        self.maxlen = 0
        self.suflen = 0
        self.__init_fixed_features()
        self.word_index = {}
        self.lc_word_index = {}
        self.suf_index = {}
        self.shape_index = {}
        self.label_index = {}
        self.gazetteer_index = {}
        self.gazetteer_phrases = {}
        legacy_gazetteer_word_type = {}

        with open(name+".idx", encoding="utf-8") as f :
            for line in f.readlines(): 
                parts = line.rstrip("\n").split("\t", maxsplit=2)
                if len(parts) == 1 :
                    parts = line.split(maxsplit=2)
                (t,k,i) = parts
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)                
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'LCWORD': self.lc_word_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'SHAPE': self.shape_index[k] = int(i)
                elif t == 'LABEL': self.label_index[k] = int(i)
                elif t == 'GAZLABEL': self.gazetteer_index[k] = int(i)
                elif t == 'GAZWORD': legacy_gazetteer_word_type[k] = i
                elif t == 'GAZPHRASE': self.gazetteer_phrases[tuple(k.split(self.GAZETTEER_JOINER))] = i

        if not self.lc_word_index :
            self.lc_word_index = {'PAD': 0, 'UNK': 1}
        if not self.shape_index :
            self.shape_index = {'PAD': 0, 'UNK': 1}
        if not self.gazetteer_index :
            self.gazetteer_index = {'PAD': 0, 'NONE': 1}
        if not self.gazetteer_phrases and legacy_gazetteer_word_type :
            self.gazetteer_phrases = {(word,): label for word, label in legacy_gazetteer_word_type.items()}
        self.__rebuild_gazetteer_lookup()
                            
    
    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w", encoding="utf-8") as f :
            print(f'MAXLEN\t{self.maxlen}\t-', file=f)
            print(f'SUFLEN\t{self.suflen}\t-', file=f)
            for key in self.label_index : print(f'LABEL\t{key}\t{self.label_index[key]}', file=f)
            for key in self.word_index : print(f'WORD\t{key}\t{self.word_index[key]}', file=f)
            for key in self.lc_word_index : print(f'LCWORD\t{key}\t{self.lc_word_index[key]}', file=f)
            for key in self.suf_index : print(f'SUF\t{key}\t{self.suf_index[key]}', file=f)
            for key in self.shape_index : print(f'SHAPE\t{key}\t{self.shape_index[key]}', file=f)
            for key in self.gazetteer_index : print(f'GAZLABEL\t{key}\t{self.gazetteer_index[key]}', file=f)
            for key in self.gazetteer_phrases :
                phrase = self.GAZETTEER_JOINER.join(key)
                print(f'GAZPHRASE\t{phrase}\t{self.gazetteer_phrases[key]}', file=f)


    ## --------- encode X from given data ----------- 
    def encode_words(self, data) :        
        sentences = list(data.sentences())

        # encode and pad sentence words
        Xw = self.__encode_and_pad(
            sentences,
            lambda w: self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'],
            self.word_index['PAD']
        )
        # encode and pad lowercased words
        Xlw = self.__encode_and_pad(
            sentences,
            lambda w: self.lc_word_index[w['lc_form']] if w['lc_form'] in self.lc_word_index else self.lc_word_index['UNK'],
            self.lc_word_index['PAD']
        )
        # encode and pad suffixes
        Xs = self.__encode_and_pad(
            sentences,
            lambda w: self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index else self.suf_index['UNK'],
            self.suf_index['PAD']
        )
        # encode and pad orthographic features
        Xc = self.__encode_and_pad(
            sentences,
            lambda w: self.cap_index[self.__get_caps_feature(w['form'])],
            self.cap_index['PAD']
        )
        Xn = self.__encode_and_pad(
            sentences,
            lambda w: self.num_index[self.__get_num_feature(w['form'])],
            self.num_index['PAD']
        )
        Xd = self.__encode_and_pad(
            sentences,
            lambda w: self.dash_index[self.__get_dash_feature(w['form'])],
            self.dash_index['PAD']
        )
        Xsh = self.__encode_and_pad(
            sentences,
            lambda w: self.shape_index[self.__get_shape(w['form'])] if self.__get_shape(w['form']) in self.shape_index else self.shape_index['UNK'],
            self.shape_index['PAD']
        )
        Xg = pad_sequences(
            maxlen=self.maxlen,
            sequences=[self.__encode_gazetteer_sentence(s) for s in sentences],
            padding="post",
            value=self.gazetteer_index['PAD']
        )
        # return encoded sequences
        return [Xw, Xlw, Xs, Xc, Xn, Xd, Xsh, Xg]

    
    ## --------- encode Y from given data ----------- 
    def encode_labels(self, data) :
        # encode and pad sentence labels 
        Y = [[self.label_index[w['tag']] for w in s] for s in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)
    ## -------- get lowercase word index size ---------
    def get_n_lc_words(self) :
        return len(self.lc_word_index)
    ## -------- get suf index size ---------
    def get_n_sufs(self) :
        return len(self.suf_index)
    ## -------- get token shape index size ---------
    def get_n_shapes(self) :
        return len(self.shape_index)
    ## -------- get capitalization feature size ---------
    def get_n_caps(self) :
        return len(self.cap_index)
    ## -------- get number feature size ---------
    def get_n_nums(self) :
        return len(self.num_index)
    ## -------- get dash feature size ---------
    def get_n_dashes(self) :
        return len(self.dash_index)
    ## -------- get gazetteer feature size ---------
    def get_n_gazetteer(self) :
        return len(self.gazetteer_index)
    ## -------- get label index size ---------
    def get_n_labels(self) :
        return len(self.label_index)

    ## -------- get index for given word ---------
    def word2idx(self, w) :
        return self.word_index[w]
    ## -------- get index for given lowercased word ---------
    def lcword2idx(self, w) :
        return self.lc_word_index[w]
    ## -------- get index for given suffix --------
    def suff2idx(self, s) :
        return self.suf_index[s]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError
