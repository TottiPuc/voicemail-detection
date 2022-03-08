from flask import Flask, request,jsonify
import joblib
import librosa
import glob
import numpy as np


app = Flask(__name__)

path_dataset= '/home/dayan/Documents/VOZY/datasets/Christian-20220228T234546Z-001/Christian/prueba/'
path_models= '/home/dayan/Documents/VOZY/voicemial/modelos/'
loaded_model = joblib.load(path_models+"lr-20220303T2236.pkl")
print(loaded_model)

@app.route("/prediccion", methods=['POST'])
def predictFromFile():
  if request.method=='POST':
    audio = request.files['audio']
    print(audio)
    X, sample_rate = librosa.load(audio, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    X = [mfccs]
    out = loaded_model.predict(X)
    if out[0] == 0:
      response = {'audio': audio.filename, 'status': 'voicemail', 'accuracy':loaded_model.predict_proba(X)[0][0]}
      return jsonify(response)
    else:
      response = {'audio': audio.filename, 'status': 'no voicemail', 'accuracy':loaded_model.predict_proba(X)[0][1]}
      return jsonify(response)


if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)