import librosa
import librosa.display
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

axes = []


def plot_specgram(file_path):

    for m, path in enumerate(file_path):
        fig = plt.figure(figsize=(10, 5))
        for n, file in enumerate(os.listdir(path)):
            y, sr = librosa.load(path + file)
            S = librosa.feature.melspectrogram(y=y,\
                                               sr=sr, n_mels=128, fmax=8000)
            axes.append(fig.add_subplot(4, 5, n + 1))
            librosa.display.specshow(librosa.power_to_db(S, \
                                                         ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title(file_path[m].split("/")[-2])
        plt.tight_layout()
        plt.show()


class Preprocesamiento():

    def __init__(self, path_data):
        self.path_data = path_data
        self.df = pd.DataFrame(
            columns=['file_path', 'mfccs_40', 'chroma', 'mel', \
                     'contrast', 'tonnetz', 'duration', 'label'])
        self.df = self.df.fillna(0)  # with 0s rather than NaNs
        print('iniciando preprocesamiento de audios con caracteristicas almacenadas en {}'\
              .format(path_data))

    def process(self, file, label):
        try:
            print("processing {}".format(file))

            # here kaiser_fast is a technique used for faster extraction
            X, sample_rate = librosa.load(file, res_type='kaiser_fast')
            duration = librosa.get_duration(y=X, sr=sample_rate)
            stft = np.abs(librosa.stft(X))
            mfccs_40 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128, fmax=8000).T, axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                                      sr=sample_rate).T, axis=0)

            row = pd.Series(
                {'file_path': file, 'mfccs_40': mfccs_40, 'chroma': chroma, 'mel': mel, 'contrast': contrast,
                 'tonnetz': tonnetz, 'duration': duration, 'label': label}, name=7)
            self.df.loc[len(self.df)] = row

        except Exception as e:
            print("Error encountered while parsing file: ", e)

    def crear_dataset(self, path, label, max=100):
        index = 0
        for file in glob.glob(path):
            if index <= max:
                self.process(file, label)
                index = index + 1
        self.df.dropna(inplace=True)
        self.df.to_pickle(self.path_data + 'voicemail_dataset_train.pkl')

    def one_hot_encoding(self):
        le = preprocessing.LabelEncoder()
        le.fit(self.df['label'])
        self.df['label'] = le.transform(self.df['label'])

        print("count:")
        print(self.df.label.value_counts())
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(le_name_mapping)
        class_names = le.classes_
        self.df.to_pickle(self.path_data + 'voicemail_dataset_train_ohe.pkl')
        return class_names
