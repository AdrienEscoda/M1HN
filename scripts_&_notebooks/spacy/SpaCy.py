import glob
import spacy
import os
from os import listdir
from os.path import splitext
from os.path import basename
import pathlib
import re
import collections
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from wordcloud import WordCloud
from gensim.models import Word2Vec
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk import BigramCollocationFinder

#fonction permettant de retirer l'extension au nom de mes documents
def remove_ext(filename):
    name, extension = splitext(filename)
    return name

#fonction permettant d'enlever du nom du document les informations superflues
def filename(file):
    filename = remove_ext(file)
    name = basename(filename)
    name = name.replace('lematin_1887', '')
    name = re.sub(r'(\d{2})(\d{2})', r'\1-\2',name)
    return name

#import de l'outil de tokenisation et lemmatisation SpaCy avec reconnaissance d'entité nommées
nlp = spacy.load('fr_core_news_md', disable=["parser"])
nlp.pipeline
nlp.max_lenght = 13000000

#tokénisation et lemmatisation du corpus avec SpaCy
def nlp_corpus(dirname):
    docs = []
    files = glob.glob(os.path.join(dirname + '/*.txt'))
    for file in files:
        with(open(file, "r", encoding='utf-8')) as f:
            fichier = f.read()
            doc = nlp(fichier)
            docs.append(doc)
    return docs

#fonction qui compte le nombre de mot pour un fichier
def count_word(file):
    total_word = 0
    with(open(file, "r", encoding='utf-8')) as f:
        fichier = f.read()
        doc = nlp(fichier)
        for token in doc: 
            total_word += 1
        return total_word

#fonction qui compte le nombre de mots par fichier pour un corpus donné
def words_file(dirname):
    word={}
    files = glob.glob(os.path.join(dirname + '/*.txt'))
    for file in files:
        t = filename(file)
        n = count_word(file)
        word[t] = n
    return word

#transformation du counter en dictionnaire pour les tokens/lemmes dont la fréquence est supérieure ou égale à n
#le counter va consister à entrer le nom de la fonction choisie + le nom du dossier
def counter_to_dico(counter,n):
    dico = dict(counter)
    dico = dict((k, v) for k, v in dico.items() if v >= n)
    return dico

#retourne un nuage de mots à partir d'un dictionnaire
def word_cloud(dico):
    wc = WordCloud(background_color="white",width=1000,height=1000, max_words=100,normalize_plurals=False).generate_from_frequencies(dico)
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.imshow(wc)
    
#Décompte du nombre d'entités nommées de personnes au sein de ce corpus
def count_per(dirname):
    personnes = Counter()
    for doc in nlp_corpus(dirname):
        for ent in doc.ents:
            if ent.label_ == "PER" :
                personnes[ent.text] += 1
    return personnes

#Nombre total d'ent de personnes dans le corpus
def tot_per(dirname):
    total = 0
    personnes = Counter()
    for doc in nlp_corpus(dirname):
        for ent in doc.ents:
            if ent.label_ == "PER" :
                total += 1
    return total

#Tri du dictionnaire en fonction des valeurs par ordre décroissant
def tri_dico_decroissant(dico):
    liste = sorted(dico.items(), key=lambda kv: kv[1], reverse=True)
    sorted_dico = collections.OrderedDict(liste)
    return sorted_dico

#fonction comptant le nombre d'occurrences des entités nommées les plus fréquentes relatives au général pour un document
def count_boulanger_ent(file):
    total_boulanger = 0
    with(open(file, "r", encoding='utf-8')) as f:
        fichier = f.read()
        doc = nlp(fichier)
        for ent in doc.ents: 
            if ent.text == 'Boulanger' or ent.text == 'général Boulanger' or ent.text == 'Vive Boulanger':
                total_boulanger += 1
        return total_boulanger
    
#Décompte des occurrences des ent 'Boulanger' pour tous les documents de mon corpus dans un dictionnaire
def boulanger_ent_fichier(dirname):
    boulanger_ent = {}
    files = glob.glob(os.path.join(dirname + '/*.txt'))
    for file in files:
        t = filename(file)
        n = count_boulanger_ent(file)
        boulanger_ent[t] = n
    return boulanger_ent

#fonction comptant le nombre d'occurrences des tokens "Boulanger" pour un document
def count_boulanger_token(file):
    total_boulanger = 0
    with(open(file, "r", encoding='utf-8')) as f:
        fichier = f.read()
        doc = nlp(fichier)
        for token in doc: 
            if token.text == 'Boulanger' or token.text == 'BOULANGER':
                total_boulanger += 1
        return total_boulanger
    
#Decompte des occurrences des tokens 'Boulanger' pour tous les documents de mon corpus dans un dictionnaire  
def boulanger_token_fichier(dirname):
    boulanger_token = {}
    files = glob.glob(os.path.join(dirname + '/*.txt'))
    for file in files:
        t = filename(file)
        n = count_boulanger_token(file)
        boulanger_token[t] = n
    return boulanger_token

#Nombre de fichiers à n occurrences du token 'Boulanger' par fichier
def dico_to_decompte(dico):
    decompte = Counter(dico.values())
    liste_decompte = sorted(decompte.items(), key=lambda t: t[0])
    decompte_trie = collections.OrderedDict(liste_decompte)
    return decompte_trie

#Tri du dictionnaire par date 
def tri_dico_date(dico):
    liste = sorted(dico.items(), key=lambda t: t[0])
    sorted_dico = collections.OrderedDict(liste)
    return sorted_dico

#construction d'une liste de listes de string afin d'entraîner le modèle word2vec
def strings_to_liste(dirname):
    sentences = []
    files = glob.glob(os.path.join(dirname + '/*.txt'))
    for file in files :
        with(open(file, "r", encoding='utf-8')) as f:
            fichier = f.read()
            fichier = fichier.split()
            sentences.append(fichier)
    return sentences

#outil de lemmatisation de la librairie nltk
lemmatizer = WordNetLemmatizer()

#fonction qui lemmatize un fichier et retourne une liste de strings
def nltk_lemmatize(file):
    with(open(file, "r", encoding='utf-8')) as f:
        fichier = f.read()
        fichier = fichier.split()
        lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in fichier])
        words = list(lemmatized_output.split(" "))
    return words

#liste de listes de string comprenant mon corpus
def liste_de_listes(dirname):
    liste = []
    files = glob.glob(os.path.join(dirname + '/*.txt'))
    for file in files:
        l = nltk_lemmatize(file)
        liste.append(l)
    return liste

#code permettant de rassembler en une seule liste tous les strings de mes sous listes
#Deux fonctions différentes car la triple boucle "for" faisait planter le notebook
def flat_list(liste):
    flat_list = []
    for sublist in liste:
        for item in sublist:
            flat_list.append(item)
    return flat_list

#fonction qui permet de ressortir les n-gramsmes (nombre à définir) pour un corpus donné
def ngrams(liste,n):
    a = (pd.Series(nltk.ngrams(liste, n)).value_counts())[:10]
    return a

#code qui permet de se concentrer sur les bigrammes autour du terme "Boulanger" les plus fréquents à partir de raw_freq
def bigram_boulanger(liste):
    finder = BigramCollocationFinder.from_words(liste)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    french_stopwords = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "afin", "ah", "ai", "aie", "aient", "aies", "ait", "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune", "audit", "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres", "autrui", "aux", "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons", "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "celà", "cent", "cents", "cependant", "certain", "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci", "cinq", "cinquante", "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit", "cinquante-neuf", "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois", "cl", "cm", "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de", "depuis", "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant", "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel", "durant", "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin", "entre", "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f", "fait", "fi", "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem", "heu", "hg", "hl", "hm", "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i", "ici", "il", "ils", "j", "j'", "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'", "jusqu'au", "jusqu'aux", "jusqu'à", "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre", "l'on", "l'un", "l'une", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "lez", "lors", "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma", "maint", "mainte", "maintes", "maints", "mais", "malgré", "me", "mes", "mg", "mgr", "mil", "mille", "milliards", "millions", "ml", "mm", "mm²", "moi", "moins", "mon", "moyennant", "mt", "m²", "m³", "même", "mêmes", "n", "n'avait", "n'y", "ne", "neuf", "ni", "non", "nonante", "nonobstant", "nos", "notre", "nous", "nul", "nulle", "nº", "néanmoins", "o", "octante", "oh", "on", "ont", "onze", "or", "ou", "outre", "où", "p", "par", "par-delà", "parbleu", "parce", "parmi", "pas", "passé", "pendant", "personne", "peu", "plus", "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi", "pourtant", "pourvu", "près", "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles", "qu'il", "qu'ils", "qu'on", "quand", "quant", "quarante", "quarante-cinq", "quarante-deux", "quarante-et-un", "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept", "quarante-six", "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq", "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf", "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf", "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze", "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize", "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que", "quel", "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques", "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'", "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se", "seize", "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six", "soi", "soient", "sois", "soit", "soixante", "soixante-cinq", "soixante-deux", "soixante-dix", "soixante-dix-huit", "soixante-dix-neuf", "soixante-dix-sept", "soixante-douze", "soixante-et-onze", "soixante-et-un", "soixante-et-une", "soixante-huit", "soixante-neuf", "soixante-quatorze", "soixante-quatre", "soixante-quinze", "soixante-seize", "soixante-sept", "soixante-six", "soixante-treize", "soixante-trois", "sommes", "son", "sont", "sous", "soyez", "soyons", "suis", "suite", "sur", "sus", "t", "t'", "ta", "tacatac", "tandis", "te", "tel", "telle", "telles", "tels", "tes", "toi", "ton", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "trente-cinq", "trente-deux", "trente-et-un", "trente-huit", "trente-neuf", "trente-quatre", "trente-sept","trente-six", "trente-trois", "trois", "très", "tu", "u", "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux", "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois", "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à", "ç'", "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée", "étées", "étés", "êtes", "être", "ô"]
    finder.apply_word_filter(lambda w: w in french_stopwords)
    word_filter = lambda w1, w2: "Boulanger" not in (w1, w2)
    finder.apply_ngram_filter(word_filter)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    scored_2 = dict(scored)
    bigrams = {key:value for key, value in scored_2.items() if value > 0.0000015}
    return bigrams
    
#code qui permet de se concentrer sur les bigrammes autour du terme "Boulanger" les plus fréquents à partir de likelihood_ratio
def bigram_boulanger_lh_ratio(liste):
    finder = BigramCollocationFinder.from_words(liste)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    french_stopwords = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "afin", "ah", "ai", "aie", "aient", "aies", "ait", "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune", "audit", "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres", "autrui", "aux", "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons", "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "celà", "cent", "cents", "cependant", "certain", "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci", "cinq", "cinquante", "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit", "cinquante-neuf", "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois", "cl", "cm", "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de", "depuis", "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant", "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel", "durant", "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin", "entre", "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f", "fait", "fi", "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem", "heu", "hg", "hl", "hm", "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i", "ici", "il", "ils", "j", "j'", "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'", "jusqu'au", "jusqu'aux", "jusqu'à", "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre", "l'on", "l'un", "l'une", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "lez", "lors", "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma", "maint", "mainte", "maintes", "maints", "mais", "malgré", "me", "mes", "mg", "mgr", "mil", "mille", "milliards", "millions", "ml", "mm", "mm²", "moi", "moins", "mon", "moyennant", "mt", "m²", "m³", "même", "mêmes", "n", "n'avait", "n'y", "ne", "neuf", "ni", "non", "nonante", "nonobstant", "nos", "notre", "nous", "nul", "nulle", "nº", "néanmoins", "o", "octante", "oh", "on", "ont", "onze", "or", "ou", "outre", "où", "p", "par", "par-delà", "parbleu", "parce", "parmi", "pas", "passé", "pendant", "personne", "peu", "plus", "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi", "pourtant", "pourvu", "près", "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles", "qu'il", "qu'ils", "qu'on", "quand", "quant", "quarante", "quarante-cinq", "quarante-deux", "quarante-et-un", "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept", "quarante-six", "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq", "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf", "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf", "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze", "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize", "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que", "quel", "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques", "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'", "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se", "seize", "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six", "soi", "soient", "sois", "soit", "soixante", "soixante-cinq", "soixante-deux", "soixante-dix", "soixante-dix-huit", "soixante-dix-neuf", "soixante-dix-sept", "soixante-douze", "soixante-et-onze", "soixante-et-un", "soixante-et-une", "soixante-huit", "soixante-neuf", "soixante-quatorze", "soixante-quatre", "soixante-quinze", "soixante-seize", "soixante-sept", "soixante-six", "soixante-treize", "soixante-trois", "sommes", "son", "sont", "sous", "soyez", "soyons", "suis", "suite", "sur", "sus", "t", "t'", "ta", "tacatac", "tandis", "te", "tel", "telle", "telles", "tels", "tes", "toi", "ton", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "trente-cinq", "trente-deux", "trente-et-un", "trente-huit", "trente-neuf", "trente-quatre", "trente-sept","trente-six", "trente-trois", "trois", "très", "tu", "u", "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux", "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois", "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à", "ç'", "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée", "étées", "étés", "êtes", "être", "ô"]
    finder.apply_word_filter(lambda w: w in french_stopwords)
    word_filter = lambda w1, w2: "Boulanger" not in (w1, w2)
    finder.apply_ngram_filter(word_filter)
    scored = finder.score_ngrams(bigram_measures.likelihood_ratio)
    scored_2 = dict(scored)
    bigrams_2 = {key:value for key, value in scored_2.items() if value > 16}
    return bigrams_2