from preprocesamiento import Preprocesamiento, plot_specgram
from entrenamiento import Entrenamiento
from modelos import modelos

RECORDINGS_PATH = '/home/dayan/Documents/VOZY/datasets/Christian-20220228T234546Z-001/Christian/prueba/'
path_maquina=RECORDINGS_PATH + "maquina/"
path_gente=RECORDINGS_PATH + "gente/"
path_dataset= '/home/dayan/Documents/VOZY/voicemial/dataset/'
path_models= '/home/dayan/Documents/VOZY/voicemial/modelos/'
classi = 'lr'
plot_specgram([path_maquina,path_gente])

inicio = Preprocesamiento(path_dataset)

inicio.crear_dataset(path_maquina+"*.wav", label="beep")
inicio.crear_dataset(path_gente+"*.wav", label="speech")
print(inicio.df.label.value_counts())
class_names=inicio.one_hot_encoding()

#modelo a entrenar clasificador:
# lr=logisticRegression   solver= liblinear or solver = lbfgs
# rf= randomForest
# xgb = xgboost
# svm = suport vetor machine
'''
model = modelos(model=classi, solver='liblinear', max_iter=4000)
train = Entrenamiento(inicio.df, model)
train.train(['mfccs_40'],class_names,classi,path_models)
'''