import os
import pandas as pd
from pandas import DataFrame
import fastwer

#calcul du taux d'erreur de caractères entre chaque sortie ocr et son texte référence
def cer(file):
    df = pd.read_csv(file)
    #remplace les sauts de ligne des sorties ocr et des textes références
    #par des espaces
    df['ocr_output'] = df['ocr_output'].apply(lambda x: x.replace('\n',' '))
    df['ref_text'] = df['ref_text'].apply(lambda x: x.replace('\n',' '))
    #retire les espaces superfétatoires entre les mots coupés sur deux lignes
    #dans les textes références uniquement
    df['ref_text'] = df.ref_text.str.replace('- ', '-')
    df['cer'] = ''
    #calcul du taux d'erreur de caractères pour chaque extrait
    for index, row in df.iterrows():
        filename = row['date_extraits']
        ref = row['ref_text']
        output = row['ocr_output']
        cer = fastwer.score_sent(output, ref, char_level=True)
        df.loc[df['date_extraits'] == filename, 'cer'] = round(cer,2)
    return df

#ajoute au dataframe le total du nombre de mots et de caractères des textes de référence
def tot_char_word(df):
    #calcul du nombre de caractères total pour chaque texte de référence
    df['nb_char_reftext'] = ''
    df['nb_char_reftext'] = df['ref_text'].str.len()
    #calcul du nombre de mots pour chaque texte de référence
    df['nb_mots_reftext'] = ''
    df['ref_text'] = df['ref_text'].str.replace("'"," ")
    df['nb_mots_reftext'] = df['ref_text'].str.split().str.len()
    df['nb_mots_reftext'].index = df['nb_mots_reftext'].index.astype(str) + ' words:'
    df['nb_mots_reftext'].sort_index(inplace=True)
    return df