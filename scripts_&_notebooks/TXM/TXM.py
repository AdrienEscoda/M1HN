import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display
from wordcloud import WordCloud
import collections
from collections import Counter

#Nuage de points des cooccurrents du mot choisi
def cooc(file):
    df = pd.read_csv(file, sep = '\t')
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(12, 8)
    colors = np.where(df["Occ"]=="général",'green', np.where(df["Occ"]=="Boulanger",'r','deepskyblue'))
    ax.scatter(x=df['Indice'],y=df['CoFréq'],c=colors)
    ax.set_xlabel('Indice de spécificité')
    ax.set_ylabel('Cofréquence')
    for idx, row in df.iterrows():
        ax.annotate(row['Occ'], (row['Indice'], row['CoFréq']))
    plt.show()
    
#Nuage de lemmes en forme de nuage de mots
def nuage(file):
    df = pd.read_csv(file, sep = '\t')
    dico = df.set_index('frlemma').to_dict()['F']
    wc = WordCloud(width=800, height=400, max_words=200, background_color = "white").generate_from_frequencies(dico)
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
#Nombre de fichiers à n occurrences du token choisi par fichier
def dico_to_decompte(dico):
    decompte = Counter(dico.values())
    liste_decompte = sorted(decompte.items(), key=lambda t: t[0])
    decompte_trie = collections.OrderedDict(liste_decompte)
    return decompte_trie

#Histogramme du décompte du nombre de fichiers à n occurrences du token 'Boulanger'
def histo_boulanger(file):
    df = pd.read_csv(file)
    dico_boulanger = df.set_index('date').to_dict()['Boulanger']
    decompte_boulanger = dico_to_decompte(dico_boulanger)
    fig = plt.figure(figsize=(40, 6), dpi=80)
    fig = plt.bar(list(decompte_boulanger.keys()), decompte_boulanger.values(), color='g')
    plt.xlabel("Nombres d'occurrences")
    plt.ylabel("Nombre de quotidiens")
    plt.show()

#Histogramme du décompte du nombre de fichiers à n occurrences du token 'Bismarck'    
def histo_bismarck(file):
    df = pd.read_csv(file)
    dico_bismarck = df.set_index('date').to_dict()['Bismarck']
    decompte_bismarck = dico_to_decompte(dico_bismarck)
    fig = plt.figure(figsize=(40, 6), dpi=80)
    fig = plt.bar(list(decompte_bismarck.keys()), decompte_bismarck.values(), color='g')
    plt.xlabel("Nombres d'occurrences")
    plt.ylabel("Nombre de quotidiens")
    plt.show()

#Histogramme du décompte du nombre de fichiers à n occurrences du token 'Clémenceau'    
def histo_clemenceau(file):
    df = pd.read_csv(file)
    dico_clemenceau = df.set_index('date').to_dict()['Clémenceau']
    decompte_clemenceau = dico_to_decompte(dico_clemenceau)
    fig = plt.figure(figsize=(40, 6), dpi=80)
    fig = plt.bar(list(decompte_clemenceau.keys()), decompte_clemenceau.values(), color='g')
    plt.xlabel("Nombres d'occurrences")
    plt.ylabel("Nombre de quotidiens")
    plt.show()
    
#Histogramme du décompte du nombre de fichiers à n occurrences du token 'Goblet'
def histo_goblet(file):
    df = pd.read_csv(file)
    dico_goblet = df.set_index('date').to_dict()['Goblet']
    decompte_goblet = dico_to_decompte(dico_goblet)
    fig = plt.figure(figsize=(40, 6), dpi=80)
    fig = plt.bar(list(decompte_goblet.keys()), decompte_goblet.values(), color='g')
    plt.xlabel("Nombres d'occurrences")
    plt.ylabel("Nombre de quotidiens")
    plt.show()
    
#Histogramme du décompte du nombre de fichiers à n occurrences du token 'Grévy'    
def histo_grevy(file):
    df = pd.read_csv(file)
    dico_grevy = df.set_index('date').to_dict()['Grévy']
    decompte_grevy = dico_to_decompte(dico_grevy)
    fig = plt.figure(figsize=(40, 6), dpi=80)
    fig = plt.bar(list(decompte_grevy.keys()), decompte_grevy.values(), color='g')
    plt.xlabel("Nombres d'occurrences")
    plt.ylabel("Nombre de quotidiens")
    plt.show()