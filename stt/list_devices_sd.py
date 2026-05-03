import sounddevice as sd

def list_devices():
    print("--- SoundDevice Input Devices ---")
    print(sd.query_devices())

if __name__ == "__main__":
    list_devices()
