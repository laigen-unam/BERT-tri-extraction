import os
import pandas as pd
import re
from itertools import product
import pickle



files = 0
# Local Alfredo
#output_path_pkl = "/Users/avarela/lab_nlp/lab_gits/deep-learning-for-bionlp/Data-sets/STM_data_set_articles_for_LUKE.pkl"
# Server pakal 
output_path_pkl = "/export/storage/users/avarela/deep_learning_models/Data-sets/STM_data_set_dl_articles_4600_preprocessed_for_luke.pkl"


# Local Carlos
# articles_path = "/home/cmendezc/Documents/ccg/gitlab-deep-learning-for-bionlp/Data-sets/STM-data-set-dl-articles-preprocessed"
# Local Alfredo 
#articles_path = "/Users/avarela/lab_nlp/lab_gits/deep-learning-for-bionlp/Data-sets/STM-data-set-dl-articles-preprocessed"
# Server pakal
articles_path = "/export/storage/users/avarela/deep_learning_models/Data-sets/STM-data-set-dl-articles-4600-preprocessed"
# Local alfredo test
#articles_path = "/Users/avarela/lab_nlp/lab_gits/deep-learning-for-bionlp/Data-sets/pruebas_preprocessed"
# Local carlos test
#articles_path = "/home/cmendezc/Documents/ccg/gitlab-deep-learning-for-bionlp/Data-sets/pruebas_preprocessed"


# Local Alfredo 
#tf_list_path = "/Users/avarela/lab_nlp/lab_gits/deep-learning-for-bionlp/Data-sets/STM_tfs.tsv"
# Local Carlos 
#tf_list_path = "/home/cmendezc/Documents/ccg/gitlab-deep-learning-for-bionlp/Data-sets/STM_tfs.tsv"
# Server pakal 
tf_list_path = "/export/storage/users/avarela/deep_learning_models/Data-sets/STM_tfs.tsv" 

# Local Alfredo 
#regulated_list_path = "/Users/avarela/lab_nlp/lab_gits/deep-learning-for-bionlp/Data-sets/STM_regulated.tsv"
# Local Carlos 
#regulated_list_path = "/home/cmendezc/Documents/ccg/gitlab-deep-learning-for-bionlp/Data-sets/STM_regulated.tsv"
# Server pakal 
regulated_list_path = "/export/storage/users/avarela/deep_learning_models/Data-sets/STM_regulated.tsv"

# Local Alfredo 
#output_path = "/Users/avarela/lab_nlp/lab_gits/deep-learning-for-bionlp/Data-sets/STM_data_set_articles_for_LUKE.tsv"
# Local Carlos 
#output_path = "/home/cmendezc/Documents/ccg/gitlab-deep-learning-for-bionlp/Data-sets/STM_data_set_articles_for_LUKE.tsv"
# Server pakal 
output_path = "/export/storage/users/avarela/deep_learning_models/Data-sets/STM_data_set_dl_articles_4600_preprocessed_for_luke.tsv"

df_tf_list = pd.read_csv(tf_list_path, sep="\t")
# print(df_tf_list.head())

df_regulated_list = pd.read_csv(regulated_list_path, sep="\t")
# print(df_tf_list.head())

df_for_luke = pd.DataFrame(columns=['sentence', 'entity_spans', 'regulator', 'regulated'])
idx_sent = 0
# Para cada archivo en el directoroi de entrada
for f in os.listdir(articles_path):
    print("Archivo: {}".format(f))
    # Abre archivo
    with open(os.path.join(articles_path, f), mode="r") as ifile:
        # Para cada línea
        for line in ifile:
            spans_tf = []
            spans_regulated = []
            flag_tfs = False
            flag_regulated = False
            line = line.rstrip("\n")
            # Para cada TF de la lista
            for idx in df_tf_list.index:
                tf = df_tf_list.iloc[idx]['0']
                # Buscar TF separado por espacios
                for match in re.finditer(r' (?P<tf>' + tf + ') ', line):
                    flag_tfs = True
                    # print(match.span())
                    span = match.span()
                    print("TF {} encontrado en {}".format(tf, span))
                    print("Entidad: {}".format(line[span[0]:span[1]]))
                    spans_tf.append(list(span))
                    print(str(spans_tf))
            # Si sí se encontraron TFs
            if flag_tfs:
                # Para cada regulado en la lista
                for idx in df_regulated_list.index:
                    regulated = df_regulated_list.iloc[idx]['0']
                    # Buscar TF separado por espacios
                    for match in re.finditer(r' (?P<regulated>' + regulated + ') ', line):
                        flag_regulated = True
                        # print(match.span())
                        span = match.span()
                        print("regulated {} encontrado en {}".format(regulated, span))
                        print("Entidad: {}".format(line[span[0]:span[1]]))
                        spans_regulated.append(list(span))
                        print(str(spans_regulated))
            if flag_tfs and flag_regulated:
                entity_lists = [spans_tf, spans_regulated]
                print("entity_lists: {}".format(entity_lists))
                entity_combination = product(*entity_lists)
                for elem in entity_combination:
                    elem = [list(x)for x in elem]
                    print("Combinations: {}".format(elem))
                    df_for_luke.at[idx_sent, 'sentence'] = line
                    df_for_luke.at[idx_sent, 'entity_spans'] = elem
                    df_for_luke.at[idx_sent, 'regulator'] = line[elem[0][0]:elem[0][1]]
                    df_for_luke.at[idx_sent, 'regulated'] = line[elem[1][0]:elem[1][1]]
                    idx_sent += 1
            print("**********")
    df_for_luke.to_csv(output_path, index=False, sep="\t")
    files += 1
print("Archivos procesados: {}".format(files))

# All curated including multi regulator - mulit regulated using pandas and protocol 4 of serialization with pickle library
with open(output_path_pkl,"wb") as f:
    pickle.dump(df_for_luke,f,protocol=4)