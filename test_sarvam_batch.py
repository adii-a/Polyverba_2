import os
from dotenv import load_dotenv
from sarvamai import SarvamAI
import time

def main():
    # Load environment variables
    load_dotenv()
    
    # Take API key from .env
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        print("Error: SARVAM_API_KEY not found in .env file.")
        return

    # Initialize SarvamAI client
    client = SarvamAI(api_subscription_key=api_key)

    # Note: 'mode' can be 'transcribe' or 'translate' depending on the requirement
    print("Creating batch job for Speech-to-Text Translation...")
    job = client.speech_to_text_job.create_job(
        model="saaras:v3",
        mode="translate", # Changed to 'translate' for translation into English/Indian language
        language_code="hi-IN", # Set to expected source or target
        with_diarization=True,
        num_speakers=2
    )

    # You can update these paths to any audio files you want to batch process
    audio_paths = ["sample_audio.wav"] 
    
    # Check if files exist
    valid_paths = [p for p in audio_paths if os.path.exists(p)]
    if not valid_paths:
        print(f"No valid audio files found to upload. Please provide valid paths instead of {audio_paths}")
        return

    print(f"Uploading files: {valid_paths}")
    job.upload_files(file_paths=valid_paths)
    job.start()

    print("Waiting for job completion...")
    job.wait_until_complete()

    # Check file-level results
    file_results = job.get_file_results()

    print(f"\nSuccessful: {len(file_results['successful'])}")
    for f in file_results['successful']:
        print(f"  ✓ {f['file_name']}")

    print(f"\nFailed: {len(file_results['failed'])}")
    for f in file_results['failed']:
        print(f"  ✗ {f['file_name']}: {f['error_message']}")

    # Download outputs for successful files
    if file_results['successful']:
        os.makedirs("./output", exist_ok=True)
        job.download_outputs(output_dir="./output")
        print(f"\nDownloaded {len(file_results['successful'])} file(s) to: ./output")

if __name__ == "__main__":
    main()
