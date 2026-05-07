import sounddevice as sd
import time

def cb(indata, frames, time_info, status):
    pass

devices = sd.query_devices()
for i, d in enumerate(devices):
    if 'CABLE Output' in d['name'] and d['max_input_channels'] > 0:
        api = sd.query_hostapis(d['hostapi'])['name']
        print(f"Testing {api} - {d['name']} (ID {i})")
        
        # Test 1 channel
        try:
            with sd.InputStream(device=i, channels=1, samplerate=int(d['default_samplerate']), callback=cb):
                time.sleep(0.5)
                print("  Success with 1 channel")
        except Exception as e:
            print(f"  Failed with 1 channel: {e}")
            
        # Test 2 channels
        try:
            with sd.InputStream(device=i, channels=2, samplerate=int(d['default_samplerate']), callback=cb):
                time.sleep(0.5)
                print("  Success with 2 channels")
        except Exception as e:
            print(f"  Failed with 2 channels: {e}")

        # Test max channels
        max_ch = d['max_input_channels']
        if max_ch not in (1, 2):
            try:
                with sd.InputStream(device=i, channels=max_ch, samplerate=int(d['default_samplerate']), callback=cb):
                    time.sleep(0.5)
                    print(f"  Success with {max_ch} channels")
            except Exception as e:
                print(f"  Failed with {max_ch} channels: {e}")
