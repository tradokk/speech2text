#!/usr/bin/env python3
import pyaudio
import deepspeech
import numpy as np
from queue import SimpleQueue

BUFFERS_PER_SECOND = 10
SAMPLE_WIDTH = 2
BEAM_WIDTH = 512

#switch between tensorflow and tensorflow light model
#MODEL_PATH = 'deepspeech-0.8.1-models.tflite'
MODEL_PATH = 'deepspeech-0.8.1-models.pbmm'

SCORER_PATH = 'deepspeech-0.8.1-models.scorer'

buffer_queue = SimpleQueue()


def audio_callback(in_data, frame_count, time_info, status_flags):
    buffer_queue.put(np.frombuffer(in_data, dtype='int16'))
    return (None, pyaudio.paContinue)


def find_device(pyaudio, device_name):
    ''' find specific device or return default input device'''
    default = pyaudio.get_default_input_device_info()
    for i in range(pyaudio.get_device_count()):
        name = pyaudio.get_device_info_by_index(i)['name']
        if name == device_name:
            return (i, name)
    return (default['index'], default['name'])


def main():
    model = deepspeech.Model(MODEL_PATH)
    model.setBeamWidth(BEAM_WIDTH)
    model.enableExternalScorer(SCORER_PATH)

    stream = model.createStream()

    audio = pyaudio.PyAudio()
    index, name = find_device(audio, 'pulse')

    print(f'select device {name}')

    buffer_size = model.sampleRate() // BUFFERS_PER_SECOND
    audio_stream = audio.open(rate=model.sampleRate(),
                              channels=1,
                              format=audio.get_format_from_width(
                                  SAMPLE_WIDTH, unsigned=False),
                              input_device_index=index,
                              input=True,
                              frames_per_buffer=buffer_size,
                              stream_callback=audio_callback)

    num_iterations = BUFFERS_PER_SECOND * 2
    i = 0
    while audio_stream.is_active():
        stream.feedAudioContent(buffer_queue.get())
        if i % num_iterations == 0:
            text = stream.intermediateDecode()
            if text.find('stop') >= 0:
                break
            print(text)
        i += 1

    print(stream.finishStream())
    audio_stream.close()


if __name__ == '__main__':
    main()
