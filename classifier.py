import re
import os
import math
from string import punctuation
from pymystem3 import Mystem
from time import time
import codecs
import random


def tokenize(text):
    l = re.split(r'\s+|[' + punctuation + ']\s*', preProcess(text))
    res = list(filter(lambda e: e != '',l))
    return res

def preProcess(text):
    text = text.lower()
    return text

def lemmatize(myStem, text):
    lemmas = myStem.lemmatize(text)
    return ''.join(lemmas[:-1]) # -1 to delete \n symbol in the end

def buildDict(words, texts):
    my_dict = dict.fromkeys(words, 0)
    for t in texts:
        for w in set(t):
            my_dict[w] += 1
    return my_dict

def classWordWeight(occ, classDocCount):
    # using addOne smoothing in Bernoulli model
    return (1 + occ) / (2 + classDocCount)

def classWeight(textWords, classWeight, classDocCount):
    s = 0
    for w in classWeight:
        if w in textWords:
            s += math.log(classWeight[w])
        else:
            s += math.log(1 - classWeight[w]) 
    return s 

def predict(text, fstClassWeight, sndClassWeight, fstClassCount,
        sndClassCount, fstClassProb, sndClassProb):
    tokens = tokenize(text)
    textWords = set(tokens)
    # skip words which havent been in the train set 
    # fstClassWeight and sndClassWeight keys (words) are equal here
    textWords = [w for w in textWords if (w in fstClassWeight)]
    fstClassTextProb = classWeight(textWords, fstClassWeight, fstClassCount)
    sndClassTextProb = classWeight(textWords, sndClassWeight, sndClassCount)

    resFstProb = fstClassTextProb * fstClassProb
    resSndProb = sndClassTextProb * sndClassProb
    if (resFstProb >= resSndProb):
        return ('sport', resFstProb, resSndProb)
    else:
        return ('policy', resFstProb, resSndProb)

def count_labels(labels):
    return {label: sum(1 for l in labels if l == label)
        for label in set(labels)}

def train(train_texts, train_labels):
    label2cnt = count_labels(train_labels)
    print('Labels counts:', label2cnt)

    fstClassTexts = [tokenize(train_texts[i])
        for i in range(len(train_texts)) if train_labels[i] == 'sport']
    sndClassTexts = [tokenize(train_texts[i])
        for i in range(len(train_texts)) if train_labels[i] == 'policy']

    fstWords = [word for t in fstClassTexts for word in set(t)]
    fstClassDict = buildDict(fstWords, fstClassTexts)

    sndWords = [word for t in sndClassTexts for word in set(t)]
    sndClassDict = buildDict(sndWords, sndClassTexts)

    fstZeros = {k : 0 for k in
        set(sndClassDict.keys()).difference(set(fstClassDict.keys()))}
    sndZeros = {k : 0 for k in
        set(fstClassDict.keys()).difference(set(sndClassDict.keys()))}

    fstClassDict.update(fstZeros)
    sndClassDict.update(sndZeros)
    print("fstClassDictLen: ", len(fstClassDict))
    print("sndClassDictLen: ", len(sndClassDict))

    fstClassDict = dict(sorted(fstClassDict.items(),
        key=lambda t: t[1], reverse=True))
    sndClassDict = dict(sorted(sndClassDict.items(),
        key=lambda t: t[1], reverse=True))

    fstClassCount = len(fstClassTexts)
    sndClassCount = len(sndClassTexts) 

    fstClassWeight = dict(map(lambda e:
        (e[0], classWordWeight(e[1], fstClassCount)), fstClassDict.items()))
    sndClassWeight = dict(map(lambda e:
        (e[0], classWordWeight(e[1], sndClassCount)), sndClassDict.items()))

    fstClassProb = len(fstClassTexts) / len(fstClassTexts + sndClassTexts)
    sndClassProb = len(sndClassTexts) / len(fstClassTexts + sndClassTexts)
    return {'sport': (fstClassWeight, fstClassCount, fstClassProb),
            'policy': (sndClassWeight, sndClassCount, sndClassProb)}


def classify(texts, params):
    fstClassWeight, fstClassCount, fstClassProb = params['sport']
    sndClassWeight, sndClassCount, sndClassProb = params['policy']
    res = [predict(t, fstClassWeight, sndClassWeight, fstClassCount, sndClassCount, fstClassProb, sndClassProb) for t in texts]
    labels = [r[0] for r in res]
    print('Predicted labels counts:')
    print(count_labels(labels))
    return res

def load_data(data_dir='news', parts=('train','test')):
    """
    Loads data from specified directory. Returns dictionary part->(list of texts, list of corresponding labels).
    """
    part2xy = {} # tuple(list of texts, list of their labels) for train and test parts
    myStem = Mystem()
    for part in parts:
        print('Loading %s set ' % part)

        xpath = os.path.join(data_dir, '%s.texts' % part)
        with codecs.open(xpath, 'r', encoding='utf-8') as inp:
            wholeText = inp.read().strip()
            texts = lemmatize(myStem, wholeText).split('\n')

        ypath = os.path.join(data_dir, '%s.labels' % part)
        if os.path.exists(ypath):
            with codecs.open(ypath, 'r', encoding='utf-8') as inp:
                labels = [s.strip() for s in inp.readlines()]
            assert len(labels) == len(texts), 'Number of labels and texts differ in %s set!' % part
            for cls in set(labels):
                print(cls, sum((1 for l in labels if l == cls)))
        else:
            labels = None
            print('unlabeled', len(texts))

        part2xy[part] = (texts, labels)
    return part2xy

def score(y_pred, y_true):
    assert len(y_pred)==len(y_true), 'Received %d but expected %d labels' % (len(y_pred), len(y_true))
    correct = sum(y1 == y2 for y1, y2 in zip(y_pred, y_true))
    print('Number of correct/incorrect predictions: %d/%d' % (correct, len(y_pred)))
    acc = 100.0 * correct / len(y_pred)
    return acc

def main():
    part2xy = load_data('news')
    train_texts, train_labels = part2xy['train']

    print('\nTraining classifier on %d examples from train set ...' % len(train_texts))
    st = time()
    params = train(train_texts, train_labels)
    print('Classifier trained in %.2fs' % (time()-st))

    rf = open('results.txt','w')
    for part, (x, y) in part2xy.items():
        print('\nClassifying %s set with %d examples ...'
            % (part, len(x)))
        st = time()
        preds = classify(x, params)
        print('%s set classified in %.2fs' % (part, time() - st))
 
        if y is None:
            print('no labels for %s set' % part)
        else:
            labels = [pr[0] for pr in preds]
            acc = score(labels, y)
            print('\nAccuracy on the %s set with %d examples is %d%%' %
                (part, len(x), acc))
            print(part, file=rf)
            for (pred, sportScore, policyScore), y_pred in zip(preds,y):
                print("Predict: %s, Fact: %s, sportScore: %.3f, policyScore: %.3f" % (pred, y_pred, sportScore, policyScore), file = rf)
    rf.close()

if __name__=='__main__':
    main()
