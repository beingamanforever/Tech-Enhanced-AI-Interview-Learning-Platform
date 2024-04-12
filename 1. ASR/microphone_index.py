import pyaudio

pa = pyaudio.PyAudio()

for index in range(pa.get_device_count()):
    device_info = pa.get_device_info_by_index(index)
    if device_info['maxInputChannels'] > 0:
        print(f"Index: {index}, Name: {device_info['name']}")