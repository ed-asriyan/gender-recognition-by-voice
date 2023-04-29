from enum import Enum

from librosa import core, stft, effects, feature
from numpy import abs, mean, hstack, array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


def create_model(vector_length=128):
    """5 hidden dense layers from 256 units to 64, not the best model, but not bad."""
    model = Sequential()
    model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # one output neuron with sigmoid activation function, 0 means female, 1 means male
    model.add(Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's male/female classification (binary)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    # load the saved/trained weights
    model.load_weights("model.h5")
    return model


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = core.load(file_name)
    if chroma or contrast:
        stft = abs(stft(X))
    result = array([])
    if mfcc:
        mfccs = mean(feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = hstack((result, mfccs))
    if chroma:
        chroma = mean(feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = hstack((result, chroma))
    if mel:
        mel = mean(feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = hstack((result, mel))
    if contrast:
        contrast = mean(feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        result = hstack((result, contrast))
    if tonnetz:
        tonnetz = mean(feature.tonnetz(y=effects.harmonic(X), sr=sample_rate).T, axis=0)
        result = hstack((result, tonnetz))
    return result


class Gender(Enum):
    Male = 0
    Female = 1

class Recognizer:
    def __init__(self):
        self._model = create_model()

    def recognize(self, path: str) -> Gender:
        features = extract_feature(path, mel=True).reshape(1, -1)
        return Gender.Male if self._model.predict(features, verbose=0)[0][0] >= 0.5 else Gender.Female


if __name__ == "__main__":
    # load the saved model (after training)
    # model = pickle.load(open("result/mlp_classifier.model", "rb"))
    import argparse

    parser = argparse.ArgumentParser(description="Gender recognition script that returns MALE or FEMALE")
    parser.add_argument("-f", "--file", help="The path to the file, preferred to be in WAV format")
    args = parser.parse_args()
    result = "male" if Recognizer().recognize(args.file) == Gender.Male else "female"
    print(result)
