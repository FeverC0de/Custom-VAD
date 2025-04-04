import torch
import torchaudio.transforms as T
import numpy as np
from silero_vad import get_speech_timestamps, VADIterator
import gradio as gr
import time
import threading
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

listening = False
voice_detected = False
stop_listening_timer = 0

probability_list = [0]
time_stamp = 0
sensitivity = 0.5
latency = 0.1

max_samples = 512
probability_data = pd.DataFrame({"x":time_stamp, "y":probability_list})

device = torch.device('cuda')
model, utils = torch.hub.load(source= "local", repo_or_dir = 'snakers4\\silero-vad', model= 'silero_vad')
model.to(device)

#resamples Gradio audio to 16000hz
def resample(y, original_sample_rate, target_sample_rate: int = 16_000):
    return signal.resample(y, int(len(y) * target_sample_rate / original_sample_rate))

#sets audio chunk to np array
def preprocess_audio(y):
    if y is None:
        return np.zeros(512, dtype=np.float32)

    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32) / 32768.0
    y = y.squeeze()
    return y

def audio_callback(new_chunk):
    global voice_detected
    global listening
    global probability_data
    global time_stamp
    global sensitivity
    global stop_listening_timer
    if new_chunk is None:
        voice_detected = True
        return True

    original_sample_rate, audio = new_chunk

    audio = preprocess_audio(audio)
    audio = resample(audio, original_sample_rate)

    audio_tensor = torch.tensor(audio, dtype = torch.float32).to(device)

    if len(audio_tensor) < max_samples:
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, max_samples - len(audio_tensor)))

    else:
        audio_tensor = audio_tensor[:max_samples]


    with torch.no_grad():

        probability = model(audio_tensor, 16000).item()
        if probability > sensitivity:
            speech_detected = True
        else:
            speech_detected = False

        probability_data = probability_data._append({"x":time_stamp, "y":probability}, ignore_index = True)

        if len(probability_data) > 20:
            probability_data = probability_data.iloc[-20:]
        time_stamp += 1
    
    if speech_detected:
        listening = True
        voice_detected = True
        return True
        
    else:
        voice_detected = False
        return False
    
def update_plot():
    global probability_data
    global latency
    def retrieve_probability_data():
        return probability_data
            
    gr.update(x_lim=[probability_data["x"].iloc[0], probability_data["x"].iloc[-1]])

    return retrieve_probability_data()
        
def read_voice_state():
    global voice_detected
    global stop_listening_timer
    global listening
    global sensitivity
    global latency
    
    while True:
        time.sleep(latency)
        if not voice_detected:
            stop_listening_timer += 1
        else:
            stop_listening_timer = 0

        if stop_listening_timer >= 30:
            listening = False
            TimerStarted = False
    
def read_listening_state():
    global listening
    return listening

def change_sensitivity(value):  
    global sensitivity
    sensitivity = value

def change_latency(value):
    global latency
    latency = value
    gr.update(every = latency, label = str(probability_data["x"]))

def test():
    print(10)
    



end_thread = threading.Thread(target=read_voice_state)
end_thread.start()

with gr.Blocks() as ui:
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], streaming=True)
        voice_detection = gr.Textbox(label = "Voice Detected")
        listening = gr.Textbox(label = "listening", value = listening)


    plot = gr.LinePlot(update_plot, label = "Probability",  x = "x", y = "y", every= latency, y_lim= [0, 1])
    sensitivity_slider = gr.Slider(minimum= 0, maximum= 1, value = sensitivity, label= "Sensitivity")
    sensitivity_slider.change(change_sensitivity, inputs= sensitivity_slider,)
    latency_slider = gr.Slider(minimum = 0, maximum= 1, value = latency, label = "Latency" )
    latency_slider.change(change_latency, inputs= [latency_slider], outputs= plot)

    t = gr.Timer(0.1, active= True)
    t.tick(fn = read_listening_state, outputs = [listening])

    audio_input.stream(
        fn=audio_callback,
        inputs=[audio_input],
        outputs=[voice_detection],
        stream_every= latency
    )
    
    

ui.launch(inbrowser=True)

