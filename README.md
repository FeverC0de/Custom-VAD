# Voice Activity Detection (VAD) using Silero VAD

This repository contains a Voice Activity Detection (VAD) system built using the Silero VAD model, and Gradio for a real-time interactive interface. The system detects voice activity from a microphone input and plots the probability of speech presence over time with adjustable sensitivity.

## Features
- **Real-time voice activity detection** using Silero VAD.
- **Adjustable sensitivity and latency** settings.
- **Interactive Gradio UI** for easy monitoring.
- **Plots probability data** of detected speech provided the model.

## Bugs
- **Latency slider** doesn't work at the moment will work on fixing it.
- If you wish to fix the latency, change the ```latency``` variable in **vad.py**

## To Do
- Work on changing the latency through gradio UI


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vad-project.git
   cd vad-project
   ```
2. Install dependencies:
   ```bash
   pip install torch torchaudio gradio pandas scipy matplotlib
   ```
3. Download the Silero VAD model:
   ```python
   import torch
   model, utils = torch.hub.load(source='local', repo_or_dir='snakers4/silero-vad', model='silero_vad')
   ```

## Usage

Run the application:
```bash
python VAD.py
```
This will launch a Gradio interface in your web browser.

## UI Components
- **Microphone Input**: Streams real-time audio.
- **Voice Detection Status**: Displays whether speech is detected.
- **Listening State**: Indicates if the system is actively listening.
- **Probability Plot**: Shows real-time probability of speech.
- **Sensitivity & Latency Sliders**: Adjust detection threshold and response time.

## Configuration
- **Sensitivity**: Adjust the threshold for detecting speech.
- **Latency**: Control the update interval for probability calculations.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [Silero VAD](https://github.com/snakers4/silero-vad)
- PyTorch, Gradio, and other open-source dependencies

