import joblib
import librosa
import glob
import numpy as np

path_dataset= '/home/dayan/Documents/VOZY/datasets/Christian-20220228T234546Z-001/Christian/prueba/'
path_models= '/home/dayan/Documents/VOZY/voicemial/modelos/'
# loaded_model = pickle.load(open("rf-20190125T2102.pkl", "rb"))
loaded_model = joblib.load(path_models+"lr-20220303T2236.pkl")
print(loaded_model)

def predictFromFile(file):
  print(file)
  X, sample_rate = librosa.load(file, res_type='kaiser_fast')
  mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
  X = [mfccs]
  out = loaded_model.predict(X)
  if out[0] == 0:
    print("prediccion ==>  El audio es un voicemail, con una precision del {}".format(loaded_model.predict_proba(X)[0][0]))
  else:
    print("prediccion ==>  El audio es de una persona, con una precision del {}".format(loaded_model.predict_proba(X)[0][1]))


for file in glob.glob(path_dataset + "test/*"):
    predictFromFile(file)