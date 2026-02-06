"""
UAMS Parkinson's Voice Feature Extractor with CSV metadata support
Based on UAMS PD_Parselmouth_mPower.ipynb
"""

import os
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
import warnings
import statistics
warnings.filterwarnings('ignore')

class UAMSFeatureExtractor:
    """
    UAMS Parkinson's Voice Feature Extractor with CSV metadata support
    """
    
    def __init__(self, target_sr=8000, segment_duration=1.5):
        """
        Initialize the extractor.
        
        Parameters:
        -----------
        target_sr : int
            Target sampling rate (8000 Hz for telephone recordings)
        segment_duration : float
            Duration to trim audio to (1.5 seconds)
        """
        self.target_sr = target_sr
        self.segment_duration = segment_duration
        
        # Gender-specific pitch ranges (from UAMS code)
        self.female_pitch_range = (100, 600)  # Hz
        self.male_pitch_range = (75, 300)     # Hz
    
    def extract_features_with_metadata(self, audio_path, gender=None, label=None, age=None, sample_id=None):
        """
        Extract features with optional metadata.
        
        Parameters:
        -----------
        audio_path : str
            Path to audio file
        gender : str or None
            'M' or 'F' or None (for auto-detect)
        label : int or str or None
            0/1 or 'HC'/'PD' or None
        age : int or None
            Age of speaker
        sample_id : str or None
            Sample ID from CSV
        
        Returns:
        --------
        features_dict : dict
            Dictionary with all features and metadata
        """
        # Preprocess audio
        audio, sr = self._preprocess_audio(audio_path)
        
        # Determine gender for pitch range
        if gender:
            is_female = (gender.upper() == 'F')
        else:
            is_female = self._detect_gender(audio, sr)
        
        # Set pitch range
        f0min, f0max = self.female_pitch_range if is_female else self.male_pitch_range
        
        # Extract acoustic features
        acoustic_features = self._extract_acoustic_features(audio, sr, f0min, f0max)
        
        # Add metadata
        acoustic_features['filename'] = os.path.basename(audio_path)
        
        if sample_id:
            acoustic_features['sample_id'] = sample_id
        
        if gender:
            acoustic_features['gender'] = gender.upper()
            acoustic_features['is_female'] = is_female
        
        if label is not None:
            # Convert label to 0/1
            if isinstance(label, str):
                label = 0 if label.upper() == 'HC' else 1
            acoustic_features['label'] = label
            acoustic_features['group'] = 'HC' if label == 0 else 'PD'
        
        if age is not None:
            acoustic_features['age'] = int(age)
        
        acoustic_features['f0min'] = f0min
        acoustic_features['f0max'] = f0max
        
        return acoustic_features
    
    def _preprocess_audio(self, audio_path):
        """Preprocess audio file."""
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        # Trim to 1.5 seconds
        target_samples = int(self.segment_duration * sr)
        if len(audio) > target_samples:
            start = (len(audio) - target_samples) // 2
            audio = audio[start:start + target_samples]
        elif len(audio) < target_samples:
            padding = target_samples - len(audio)
            audio = np.pad(audio, (padding//2, padding - padding//2), mode='constant')
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio, sr
    
    def _detect_gender(self, audio, sr):
        """Auto-detect gender from pitch."""
        try:
            sound = parselmouth.Sound(audio, sr)
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]
            
            if len(pitch_values) == 0:
                return True  # Default to female
            
            mean_pitch = np.mean(pitch_values)
            return mean_pitch > 165  # Above 165 Hz = likely female
        except:
            return True
    
    def _extract_acoustic_features(self, audio, sr, f0min, f0max):
        """Extract 23 acoustic features using Parselmouth."""
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        
        # Duration
        duration = call(sound, "Get total duration")
        
        # Pitch
        pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
        meanF0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdevF0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        
        # HNR
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        # Jitter
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Shimmer
        localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # Formants
        f1_mean, f2_mean, f3_mean, f4_mean, f1_stdev, f2_stdev, f3_stdev, f4_stdev = self._extract_formants(sound, f0min, f0max)
        
        return {
            'duration': duration,
            'meanF0Hz': meanF0,
            'stdevF0Hz': stdevF0,
            'HNR': hnr,
            'localJitter': localJitter,
            'localabsoluteJitter': localabsoluteJitter,
            'rapJitter': rapJitter,
            'ppq5Jitter': ppq5Jitter,
            'ddpJitter': ddpJitter,
            'localShimmer': localShimmer,
            'localdbShimmer': localdbShimmer,
            'apq3Shimmer': apq3Shimmer,
            'apq5Shimmer': apq5Shimmer,
            'apq11Shimmer': apq11Shimmer,
            'ddaShimmer': ddaShimmer,
            'f1_mean': f1_mean,
            'f2_mean': f2_mean,
            'f3_mean': f3_mean,
            'f4_mean': f4_mean,
            'f1_stdev': f1_stdev,
            'f2_stdev': f2_stdev,
            'f3_stdev': f3_stdev,
            'f4_stdev': f4_stdev
        }
    
    def _extract_formants(self, sound, f0min, f0max):
        """Extract formants."""
        pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        
        formants = call(sound, "To Formant (burg)", 0.0025, 4, 6000, 0.025, 50)
        numPoints = call(pointProcess, "Get number of points")
        
        f1_list, f2_list, f3_list, f4_list = [], [], [], []
        
        for point in range(1, numPoints + 1):
            t = call(pointProcess, "Get time from index", point)
            f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
            f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
            f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
            f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
            
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)
        
        # Clean NaN values
        f1_list = [f for f in f1_list if not np.isnan(f)]
        f2_list = [f for f in f2_list if not np.isnan(f)]
        f3_list = [f for f in f3_list if not np.isnan(f)]
        f4_list = [f for f in f4_list if not np.isnan(f)]
        
        # Calculate statistics
        if len(f1_list) > 0:
            f1_mean = statistics.mean(f1_list)
            f2_mean = statistics.mean(f2_list)
            f3_mean = statistics.mean(f3_list)
            f4_mean = statistics.mean(f4_list)
            
            f1_stdev = statistics.stdev(f1_list) if len(f1_list) > 1 else 0
            f2_stdev = statistics.stdev(f2_list) if len(f2_list) > 1 else 0
            f3_stdev = statistics.stdev(f3_list) if len(f3_list) > 1 else 0
            f4_stdev = statistics.stdev(f4_list) if len(f4_list) > 1 else 0
        else:
            f1_mean, f2_mean, f3_mean, f4_mean = 500, 1500, 2500, 3500
            f1_stdev = f2_stdev = f3_stdev = f4_stdev = 0
        
        return f1_mean, f2_mean, f3_mean, f4_mean, f1_stdev, f2_stdev, f3_stdev, f4_stdev