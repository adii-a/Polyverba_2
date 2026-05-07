import sounddevice as sd
devices = sd.query_devices()
for i, d in enumerate(devices):
    if 'CABLE Output' in d['name'] and d['max_input_channels'] > 0:
        print(f'Testing {i} {d["name"]}')
        for ch in [1, 2]:
            try:
                with sd.InputStream(device=i, channels=ch, samplerate=int(d['default_samplerate'])):
                    print(f'  Success with {ch} channels')
            except Exception as e:
                print(f'  Failed with {ch} channels: {e}')
