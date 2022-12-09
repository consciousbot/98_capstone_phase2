"""
language translator
"""
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

import pyttsx3
import pyaudio
import sounddevice as sound
from scipy.io.wavfile import write
import wavio as wv
import time
import speech_recognition as sr
import whisper as wp
import random
import tensorflow as tf
import string
import re
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.keras.layers import Dense



model = tf.keras.models.load_model('/home/user2/Desktop/model.h5')


text_file = "/home/user2/Desktop/spa-eng/spa.txt"
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    english, spanish = line.split("\t")
    spanish = "[start] " + spanish + " [end]"
    text_pairs.append((english, spanish))

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

        

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
    lowercase, f"[{re.escape(strip_chars)}]", "")

vocab_size = 15000
sequence_length = 20

source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
    )
target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization)

train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)

batch_size = 64

def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({
        "english": eng,
        "spanish": spa[:, :-1],
        }, spa[:, 1:])

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
    print(f"targets.shape: {targets.shape}")

        

embed_dim = 256
latent_dim = 1024

source = keras.Input(shape=(None,), dtype="int64", name="english")
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
encoded_source = layers.Bidirectional(layers.GRU(latent_dim), merge_mode="sum")(x)

past_target = keras.Input(shape=(None,), dtype="int64", name="spanish")
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
decoder_gru = layers.GRU(latent_dim, return_sequences=True)
x = decoder_gru(x, initial_state=encoded_source)
x = layers.Dropout(0.5)(x)
target_next_step = layers.Dense(vocab_size, activation="softmax")(x)
seq2seq_rnn = keras.Model([source, past_target], target_next_step)

        
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])
        next_token_predictions = seq2seq_rnn.predict([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence


class trial2(toga.App):

    def startup(self):
        main_box = toga.Box()
        
        
	
        box1 = toga.Box()
        box2 = toga.Box()
        box3 = toga.Box()
        box4 = toga.Box()
        box5 = toga.Box()
        box6 = toga.Box()
        box7 = toga.Box()
        box8 = toga.Box()
        
        
        self.text_input = toga.TextInput(placeholder="Type Here")
        self.text_input.style.flex = 1
        self.text_input.style.padding_top = 50
        self.text_input.style.padding_left = 250
        
 
        box1.add(self.text_input)

        self.language_selection = toga.Selection(items=['English', 'Spanish'])
        self.language_selection.style.padding_top = 50
        self.language_selection.style.padding_left = 250
        self.language_selection.style.padding_right = 50
        

        box1.add(self.language_selection)

        translate_button = toga.Button("TRANSLATE",on_press=self.translate)
        translate_button.style.flex = 1
        translate_button.style.padding_top = 50
        translate_button.style.padding_left = 50
        translate_button.style.padding_right = 50

        box2.add(translate_button)
        
        speech_out_button = toga.Button("PLAY", on_press=self.text_to_speech)
        speech_out_button.style.padding_top = 50
        speech_out_button.style.padding_left = 50
        speech_out_button.style.padding_right = 50
        box2.add(speech_out_button)

        self.output = toga.Label("OUTPUT ---> ")
        self.output.style.flex = 1
        self.output.style.padding_top = 50
        self.output.style.padding_left= 50
        self.output.style.padding_right = 50

        box3.add(self.output)
        
        self.heading3 = toga.Label('TRANSCRIPTION')
        self.heading3.style.flex = 1
        self.heading3.style.padding_top = 50
        self.heading3.style.padding_left= 50
        self.heading3.style.padding_right = 50
        
        box4.add(self.heading3)
        
        self.audio_duration = toga.Selection(items=['10','20','30','40','50','60','70','80','90','100','1000'])
        """
        start_audio.style.flex = 1
        
        
        start_audio.style.padding_right = 50
        
        """
        self.audio_duration.style.padding_left= 50
        self.audio_duration.style.padding_top = 25
        
        box5.add(self.audio_duration)
        
        
        stop_audio = toga.Button('START', on_press=self.speech_to_text)
        """
        stop_audio.style.flex = 1
        """
        stop_audio.style.padding_top = 25
        
        stop_audio.style.padding_left= 500
        #stop_audio.padding_right = 50
        
        
        box5.add(stop_audio)
        
        self.heading4 = toga.Label('TRANSCRIPT --->   ')
        self.heading4.style.flex = 1
        self.heading4.style.padding_top = 50
        self.heading4.style.padding_left= 50
        self.heading4.style.padding_right = 50
        
        box6.add(self.heading4)
        
        translate_button2 = toga.Button("TRANSLATE", on_press=self.transcript_translate)
        translate_button2.style.flex = 1
        translate_button2.style.padding_top = 50
        translate_button2.style.padding_left = 50
        translate_button2.style.padding_right = 50
        box7.add(translate_button2)
        
        speech_out_button_1 = toga.Button("PLAY", on_press=self.text_to_speech_1)
        speech_out_button_1.style.padding_top = 50
        speech_out_button_1.style.padding_left = 50
        speech_out_button_1.style.padding_right = 50
        box7.add(speech_out_button_1)
        
        self.heading5 = toga.Label('TRANSLATED OUTPUT --->  ')
        self.heading5.style.flex = 1
        self.heading5.style.padding_top = 50
        self.heading5.style.padding_left= 50
        self.heading5.style.padding_right = 50
        
        box8.add(self.heading5)

        
        main_box.add(box1)
        main_box.add(box2)
        main_box.add(box3)
        main_box.add(box4)
        main_box.add(box5)
        main_box.add(box6)
        main_box.add(box7)
        main_box.add(box8)

        main_box.style.update(direction = COLUMN)

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    
        
    def text_to_speech(self, widget):
        engine = pyttsx3.init()
        engine.say(self.output.text)
        engine.runAndWait()

    def text_to_speech_1(self, widget):
        engine = pyttsx3.init()
        engine.say(self.heading4.text)
        engine.runAndWait()

    def speech_to_text1(self, widget):
        x = int(self.audio_duration.value)
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio_data = r.record(source, duration=x)
            text = r.recognize_google(audio_data)
            self.heading4.text = "---------OUTPUT--------"+"\n"+text

    

    def speech_to_text(self, widget):
        x = int(self.audio_duration.value)
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio_data = r.record(source, duration=x)
            """
            x = time.ctime()
            y = x.split(' ')
            y = ''.join(y)
            z = y+".wav"
            """
            with open('speech.wav', 'wb') as f:
                f.write(audio_data.get_wav_data())
                model = wp.load_model('base')
                result = model.transcribe('speech.wav')
                self.heading4.text = "---------TRANSCRIPTION---------"+"\n"+result['text']
                
    def transcript_translate(self, widget):
        model = wp.load_model('base')
        result1 = model.transcribe('speech.wav',task='translate')
        self.heading5.text = "---------TRANSLATION---------"+"\n"+result1['text']


    

    def translate(self, widget):
    	x = str(decode_sequence(self.text_input.value))
    	self.output.text = "OUTPUT-->" + x
    	
    	
    	
def main():
    return trial2()
