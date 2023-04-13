from gtts import gTTS
import random
import time
import playsound
import speech_recognition as sr
from sklearn.feature_extraction.text import CountVectorizer
import json

from sklearn.neural_network import MLPClassifier

with open("big_bot_config.json", "r") as config_file:
    data = json.load(config_file)

INTENTS = data["intents"]

X = []
y = []
for intent in INTENTS:
    examples = INTENTS[intent]["examples"]
    for example in examples:
        if len(example) < 3:
            continue
        y.append(intent)
        X.append(intent)

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

clf = MLPClassifier(random_state=2, max_iter=200)
clf.fit(X_vectorized, y)


def classify_intent(message):
    message_vectorized = vectorizer.transform([message])
    predicted_response = clf.predict(message_vectorized)
    return predicted_response[0]


def listen_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        return r.recognize_google(audio, language="ru")
    except sr.UnknownValueError:
        return "Не удалось распознать команду"
    except sr.RequestError:
        return "Не удалось выполнить запрос к Google сервису"


def do_this_command(message):
    m = message.lower()
    pred = classify_intent(m)
    res = random.choice(INTENTS[pred]["responses"])
    say_message(res)


def say_message(message):
    voice = gTTS(message, lang='ru')
    file_voice_name = "_audio_" + str(time.time()) + "_" + str(random.randint(0, 100000)) + ".mp3"
    voice.save(file_voice_name)
    playsound.playsound(file_voice_name)
    print("голосовой ассистент:" + message)


if __name__ == '__main__':
    while True:
        command = listen_command()
        print(command, "listen_command")
        do_this_command(command)
