import os
import torch
import numpy as np
from collections import namedtuple

# --- torchaudio compatibility shim for pyannote.audio 3.x on torchaudio 2.11+ nightly ---
# Multiple APIs were removed from the public namespace in nightly builds.
import torchaudio as _ta

if not hasattr(_ta, 'AudioMetaData'):
    _ta.AudioMetaData = namedtuple(
        'AudioMetaData', ['sample_rate', 'num_frames', 'num_channels', 'bits_per_sample', 'encoding']
    )
if not hasattr(_ta, 'set_audio_backend'):
    _ta.set_audio_backend = lambda *a, **kw: None
if not hasattr(_ta, 'list_audio_backends'):
    _ta.list_audio_backends = lambda: ['soundfile']
if not hasattr(_ta, 'info'):
    def _torchaudio_info_stub(filepath, **kwargs):
        import soundfile as sf
        info = sf.info(filepath)
        return _ta.AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=16,
            encoding='PCM_S'
        )
    _ta.info = _torchaudio_info_stub

# torch 2.6+ changed torch.load default to weights_only=True, breaking pyannote checkpoints.
# Restore pre-2.6 default for callers that don't specify weights_only explicitly.
import functools as _functools
_orig_torch_load = torch.load
@_functools.wraps(_orig_torch_load)
def _torch_load_compat(*args, **kwargs):
    kwargs['weights_only'] = False  # force: lightning/pyannote explicitly passes True on newer torch
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat
# -----------------------------------------------------------------------------------------

from pyannote.audio import Pipeline

class PyannoteDiarizer:
    def __init__(self):
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("WARNING: HF_TOKEN not found in environment variables. Pyannote requires this for authentication.")
        
        print("Loading Pyannote Diarization Pipeline (this may take a moment to download models)...")
        try:
            # Pyannote/HuggingFace automatically uses the HF_TOKEN environment variable.
            # Passing use_auth_token explicitly raises an error in newer versions.
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            
            # Send to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pipeline.to(device)
            print(f"Pyannote pipeline loaded successfully on {device}")
        except Exception as e:
            print(f"Failed to load Pyannote pipeline: {e}")
            self.pipeline = None

    def diarize(self, audio_data, sample_rate):
        """
        Diarizes a numpy array of audio.
        Returns a list of dicts: [{'start': float, 'end': float, 'speaker': str}, ...]
        """
        if not self.pipeline:
            return []
            
        # Pyannote expects shape (channels, samples)
        tensor = torch.from_numpy(audio_data).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
            
        try:
            diarization = self.pipeline({"waveform": tensor, "sample_rate": sample_rate})
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            return segments
        except Exception as e:
            print(f"Diarization error: {e}")
            return []
