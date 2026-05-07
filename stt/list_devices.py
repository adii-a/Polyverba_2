import soundcard as sc

def list_devices():
    print("--- Input Devices ---")
    devices = sc.all_microphones(include_loopback=True)
    for i, dev in enumerate(devices):
        print(f"{i}: {dev.name} (Loopback: {dev.isloopback})")

if __name__ == "__main__":
    list_devices()
