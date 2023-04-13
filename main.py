import datetime as dt
import json
import queue
import sys
import pyttsx3
import sounddevice as sd
import vosk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from vosk import KaldiRecognizer
import webbrowser
import words


q = queue.Queue()

engine = pyttsx3.init()
engine.setProperty('rate', 180)


def recognize(data, vectorizer, clf):
    trg = words.TRIGGERS.intersection(data.split())
    if not trg:
        return
    data.replace(list(trg)[0], '')
    text_vector = vectorizer.transform([data]).toarray()[0]
    answer = clf.predict([text_vector])[0]
    print(answer)
    func = answer.split()[0]
    speacker(answer.replace(func, ''))
    exec(func+'()')


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def main():
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(list(words.data_set.keys()))

    clf = LogisticRegression()
    clf.fit(vectors, list(words.data_set.values()))

    del words.data_set

    model = vosk.Model('small_model')
    device = sd.default.device = 0, 4
    samplerate = int(sd.query_devices(device[0], 'input')['default_samplerate'])

    try:
        with sd.RawInputStream(samplerate=samplerate, blocksize=16000, device=device[0],
                               dtype="int16", channels=1, callback=callback):

            rec = KaldiRecognizer(model, samplerate)
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    data = json.loads(rec.Result())['text']
                    recognize(data, vectorizer, clf)
                    print(data)
                else:
                    print(rec.PartialResult())

    except KeyboardInterrupt:
        print("\nDone")


def speacker(text):
    engine.say(text)
    engine.runAndWait()


def browser():
    webbrowser.open('https://www.youtube.com', new=2)


def weather():
    webbrowser.open('https://yandex.ru/pogoda/moscow', new=2)


def hello():
    speacker('здравствуйте')


def search():
    webbrowser.open('https://ya.ru', new=2)


def current_time():
    speacker(str(dt.datetime.now().hour) + ":" + str(dt.datetime.now().minute))


if __name__ == '__main__':
    main()
