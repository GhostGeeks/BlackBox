
uap3-generator.py

Page
1
/
1
100%
#!/usr/bin/env python3
"""
UAP3 WAV Generator
-----------------
Generates a 1-hour WAV file with the multi-layered audio signal.

This script creates a WAV file containing 6 distinct audio layers:
1. 7.83 Hz AM modulated over a 100 Hz carrier frequency (Schumann Resonance)
2. 528 Hz harmonic tone with added harmonics
3. 17 kHz pings every 5 seconds
4. 2.5 kHz chirps every 10 seconds
5. 432 Hz ambient pad with multiple harmonics
6. Breathing white noise shaped to simulate breathing rhythm

The output is saved as uap3_output.wav in the specified directory.
"""
import numpy as np
from scipy import signal
import wave
import time
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Audio configuration
SAMPLE_RATE = 44100  # Hz
CHANNELS = 1
DURATION = 3600  # 1 hour in seconds

# Pre-compute frequently used values
SCHUMANN_FREQ = 7.83
CARRIER_FREQ = 100
HARMONIC_BASE_FREQ = 528
AMBIENT_BASE_FREQ = 432

# Layer amplitudes
AMP_SCHUMANN = 0.15  # 7.83 Hz modulated
AMP_HARMONIC = 0.15  # 528 Hz
AMP_PING = 0.1  # 17 kHz pings
AMP_CHIRP = 0.1  # 2.5 kHz chirps
AMP_AMBIENT = 0.2  # 432 Hz ambient
AMP_BREATH = 0.15  # Breathing white noise

# Pre-calculate some common values
TWO_PI = 2 * np.pi

def generate_schumann_resonance(t):
    """Generate 7.83 Hz AM modulated over 100 Hz carrier"""
    carrier = np.sin(TWO_PI * CARRIER_FREQ * t)
    modulator = 0.5 * (1 + np.sin(TWO_PI * SCHUMANN_FREQ * t))
    return modulator * carrier * AMP_SCHUMANN

def generate_harmonic_tone(t):
    """Generate 528 Hz harmonic tone with slight variations"""
    signal = np.sin(TWO_PI * HARMONIC_BASE_FREQ * t)
    signal += 0.3 * np.sin(TWO_PI * (HARMONIC_BASE_FREQ * 2) * t)
    signal += 0.1 * np.sin(TWO_PI * (HARMONIC_BASE_FREQ * 3) * t)
    wobble = 1 + 0.001 * np.sin(TWO_PI * 0.1 * t)
    signal *= wobble
    return signal * AMP_HARMONIC

def generate_pings(t, global_time):
    """Generate 17 kHz pings every 5 seconds"""
    ping_freq = 17000
    ping_dur = 0.1  # 100ms ping
    
    # Calculate the time within the current 5-second cycle
    cycle_time = global_time % 5
    
    # Check if we're in a ping moment
    if cycle_time < ping_dur:
        # Fade in/out to avoid clicks
        envelope = np.sin(np.pi * (cycle_time / ping_dur)) ** 2
        ping = np.sin(TWO_PI * ping_freq * t) * envelope
        return ping * AMP_PING
    
    return np.zeros_like(t)

def generate_chirps(t, global_time):
    """Generate 2.5 kHz chirps every 10 seconds"""
    chirp_dur = 0.2  # 200ms chirp
    
    # Calculate the time within the current 10-second cycle
    cycle_time = global_time % 10
    
    if cycle_time < chirp_dur:
        # Create a frequency sweep (chirp)
        t_rel = np.linspace(0, chirp_dur, len(t), endpoint=False)
        sweep = signal.chirp(
            t=t_rel,
            f0=2000,  # Start frequency
            f1=3000,  # End frequency
            t1=chirp_dur,
            method='linear'
        )
        
        # Apply envelope to avoid clicks
        envelope = np.sin(np.pi * (cycle_time / chirp_dur)) ** 2
        return sweep * envelope * AMP_CHIRP
    
    return np.zeros_like(t)

def generate_ambient_pad(t):
    """Generate 432 Hz ambient pad with multiple harmonics for richness"""
    # Create a rich pad with multiple harmonics and phase variations
    pad = np.sin(TWO_PI * AMBIENT_BASE_FREQ * t)
    pad += 0.5 * np.sin(TWO_PI * (AMBIENT_BASE_FREQ * 1.5) * t + 0.3)
    pad += 0.25 * np.sin(TWO_PI * (AMBIENT_BASE_FREQ * 2) * t + 0.7)
    pad += 0.125 * np.sin(TWO_PI * (AMBIENT_BASE_FREQ * 2.5) * t + 1.1)
    
    # Add slow amplitude modulation for movement
    mod_freq = 0.1  # Modulation at 0.1 Hz (10-second cycle)
    modulator = 0.8 + 0.2 * np.sin(TWO_PI * mod_freq * t)
    
    return pad * modulator * AMP_AMBIENT

def generate_breathing_noise(t, global_time):
    """Generate white noise shaped to simulate breathing rhythm"""
    # Generate white noise
    noise = np.random.normal(0, 1, len(t))
    
    # Create breathing envelope (approx 12 breaths per minute)
    # Inhale: 2 seconds, Exhale: 3 seconds
    breath_cycle = 5  # 5 seconds per breath
    cycle_position = global_time % breath_cycle
    
    # Create envelope shape
    if cycle_position < 2:  # Inhale phase
        # Accelerating inhale
        envelope = np.sin(np.pi * cycle_position / 4) ** 2
    else:  # Exhale phase
        # Decelerating exhale
        envelope = np.cos(np.pi * (cycle_position - 2) / 6) ** 2
    
    # Apply a lowpass filter to make the noise less harsh
    b, a = signal.butter(3, 0.2)
    filtered_noise = signal.lfilter(b, a, noise)
    
    return filtered_noise * envelope * AMP_BREATH

def generate_wav_file(filename="/home/pi/uap3_output.wav", duration=DURATION):
    """Generate WAV file with the specified duration in seconds"""
    logger.info(f"Generating {duration} seconds of audio to {filename}")
    start_time = time.time()
    
    # Calculate total number of samples
    total_samples = int(duration * SAMPLE_RATE)
    
    # Process in chunks to avoid memory issues
    chunk_size = 5 * SAMPLE_RATE  # 5 seconds at a time
    chunks = total_samples // chunk_size
    remainder = total_samples % chunk_size
    
    # Create a WAV file
    with wave.open(filename, 'w') as wav_file:
        # Set WAV file parameters
        wav_file.setnchannels(CHANNELS)  # Mono
        wav_file.setsampwidth(2)         # 16-bit samples
        wav_file.setframerate(SAMPLE_RATE)
        
        # Generate and write audio in chunks
        for chunk in range(chunks + 1):
            if chunk == chunks and remainder == 0:
                break
            
            # Determine chunk duration
            if chunk < chunks:
                current_chunk_size = chunk_size
            else:
                current_chunk_size = remainder
            
            # Calculate the start time for this chunk
            chunk_start_time = chunk * chunk_size / SAMPLE_RATE
            
            # Generate time array for this chunk
            t = np.linspace(chunk_start_time, 
                          chunk_start_time + current_chunk_size/SAMPLE_RATE, 
                          current_chunk_size, endpoint=False)
            
            # Generate audio for this chunk
            minutes = int(chunk_start_time / 60)
            seconds = int(chunk_start_time % 60)
            logger.info(f"Generating chunk {chunk+1}/{chunks+1} ({minutes:02d}:{seconds:02d})...")
            
            # Generate each layer
            layer1 = generate_schumann_resonance(t)
            layer2 = generate_harmonic_tone(t)
            layer3 = np.zeros_like(t)
            layer4 = np.zeros_like(t)
            layer5 = generate_ambient_pad(t)
            layer6 = np.zeros_like(t)
            
            # Add time-dependent layers
            for i in range(0, current_chunk_size, SAMPLE_RATE // 10):  # Process in small sub-chunks
                sub_chunk_time = chunk_start_time + i / SAMPLE_RATE
                sub_chunk_size = min(SAMPLE_RATE // 10, current_chunk_size - i)
                sub_t = t[i:i+sub_chunk_size]
                
                layer3[i:i+sub_chunk_size] = generate_pings(sub_t, sub_chunk_time)
                layer4[i:i+sub_chunk_size] = generate_chirps(sub_t, sub_chunk_time)
                layer6[i:i+sub_chunk_size] = generate_breathing_noise(sub_t, sub_chunk_time)
            
            # Mix all layers
            mixed_signal = layer1 + layer2 + layer3 + layer4 + layer5 + layer6
            
            # Normalize to prevent clipping
            max_amplitude = np.max(np.abs(mixed_signal))
            if max_amplitude > 0.95:
                mixed_signal = mixed_signal * 0.95 / max_amplitude
            
            # Convert to 16-bit PCM data
            pcm_data = (mixed_signal * 32767).astype(np.int16)
            
            # Write to WAV file
            wav_file.writeframes(pcm_data.tobytes())
            
    # Calculate and display statistics
    end_time = time.time()
    elapsed_time = end_time - start_time
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    
    logger.info(f"WAV file generation complete!")
    logger.info(f"File: {filename}")
    logger.info(f"Size: {file_size_mb:.1f} MB")
    logger.info(f"Duration: {duration} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Generation took {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    return filename

if __name__ == "__main__":
    try:
        logger.info("UAP3 WAV Generator starting...")
        output_file = generate_wav_file()
        logger.info(f"Output file created: {output_file}")
        logger.info("This file can now be played by the UAP3 Player Service")
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Error during generation: {e}")
Displaying uap3-generator.py.
