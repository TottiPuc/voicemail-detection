import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools
import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

class Entrenamiento():
    
    def __init__(self, df, model):
        self.df=df
        self.model=model

        print (' iniciando el entrenamiento con la técnica {}'.format(model))

    def print_classification_results(self, y_test, res):
        print(metrics.accuracy_score(y_test, res))
        print(metrics.classification_report(y_test, res))
        print(metrics.confusion_matrix(y_test, res))
        print("")

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
    def save_model(self, model, model_name):
        filename = "{}-{}.pkl".format(model_name, datetime.datetime.now().strftime("%Y%m%dT%H%M"))
        pickle.dump(model, open(filename, 'wb'))
        #files.download(filename)

    def generateFeaturesLabels(self, features_list):
        total_features_len = np.sum([len(self.df[feature][0]) for feature in features_list])
        print("total number of features", total_features_len)
        features, labels = np.empty((0, total_features_len)), np.empty(0)
        for index, row in self.df.iterrows():
            a = []
            for feature in features_list:
                a.append(row[feature])

            features = np.vstack([features, np.hstack(a)])
            labels = np.append(labels, row["label"])
        return np.array(features), np.array(labels, dtype=np.int)

    def train(self, features, class_names,classi,path_models):
        X, y = self.generateFeaturesLabels(features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        self.model.fit(X_train, y_train)
        filename= path_models +"{}-{}.pkl".format(classi,datetime.datetime.now().strftime("%Y%m%dT%H%M"))
        pickle.dump(self.model, open(filename, 'wb'))
        print("Score:", self.model.score(X_test, y_test))

        cross_val_scores = cross_val_score(self.model, X, y, cv=5, scoring='f1_macro')
        print("cross_val_scores:", cross_val_scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))

        predictions = self.model.predict(X_test)

        self.print_classification_results(y_test, predictions)

        cm = metrics.confusion_matrix(y_test, predictions)
        self.plot_confusion_matrix(cm, class_names)

        print("*** Scaled ***")
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_transformed = scaler.transform(X_train)
        scaled_model = self.model.fit(X_train_transformed, y_train)
        X_test_transformed = scaler.transform(X_test)
        print("scaled_model score:", self.model.score(X_test_transformed, y_test))

        return self.model
 
 






