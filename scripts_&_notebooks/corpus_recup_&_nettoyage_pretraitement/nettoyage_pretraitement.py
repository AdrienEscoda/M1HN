import re
import string
import glob
import spacy
import os
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
final_stopwords_list = list(fr_stop)

def nettoyage(dirname):
    files = glob.glob(os.path.join(dirname + '/*.txt'))
    final_stopwords_list = list(fr_stop)
    #pattern de suppression de la ponctuation
    remove = string.punctuation
    remove = remove.replace("'", "") # don't remove hyphens
    pattern = r"[{}]".format(remove)
    for doc in files :
        my_file = open(doc, encoding='utf-8')
        string_list = my_file.read()
        my_file.close()
        my_file = open(doc, "w")
        #suppression manuelle des "MM." et "M." qui ne sont pas contenus dans la liste des stop-words
        string_list = re.sub(r'M\.','', string_list)
        string_list = re.sub(r'MM\.','', string_list)
        remove2 = '|'.join(final_stopwords_list)
        #retire les mots qui correspondent à ceux de la liste des stop-words
        #quelle que soit la taille du caractère
        regex = re.compile(r'\b('+remove2+r')\b', flags=re.IGNORECASE)
        string_list = regex.sub(" ", string_list)
        #suppression de la ponctuation
        string_list = re.sub(pattern,'',string_list)
        #suppression de la ponctuation non prise en compte
        string_list = re.sub(r'[»«·°• ̃ ^]',' ', string_list)
        #retire les chiffres
        string_list = re.sub(r'[0-9]+', '', string_list)
        #suppression des backslash
        string_list = re.sub(r"\\",'', string_list)
        string_list = re.sub(r"['\n]",' ', string_list)
        string_list = re.sub(" +",' ', string_list)
        my_file.write(string_list)
        my_file.close()
#problème avec les tirets : ils servent à la fois à raccorder un même mot sur deux lignes (ex : sou-\nvenir) OU
#à séparer deux tokens (ex : peut-être): comme j'ai beaucoup plus de cas où cela raccorde un même mot, j'ai fait le choix de 
#les supprimer sans rajouter d'espace, même si cela fait tâche dans certains cas