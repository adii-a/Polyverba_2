try:
    print("Importing stt.system_audio...")
    import stt.system_audio
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
