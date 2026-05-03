import sys
import time
import stt.system_audio as sa

def main():
    success = sa.start_transcription("en", "en", "base", False)
    print("Started:", success)
    for _ in range(10):
        time.sleep(1)
        print("Waiting...")

if __name__ == "__main__":
    main()
