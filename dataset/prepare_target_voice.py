from datasets import load_dataset
import os
import logging
from pathlib import Path
import soundfile as sf
import numpy as np
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_audio_file(signal: np.ndarray, sampling_rate: int, save_path: str, filename: str) -> bool:
    """
    Save an audio signal to a WAV file.
    
    Args:
        signal: The audio signal to save
        sampling_rate: The sampling rate of the audio
        save_path: Directory where the file should be saved
        filename: Name of the file (without extension)
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        audio_path = os.path.join(save_path, f"{filename}.wav")
        sf.write(audio_path, signal, sampling_rate)
        logger.debug(f"Successfully saved audio to {audio_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save audio file {filename}: {str(e)}")
        return False

        
def main():
    # Load dataset
    logger.info("Loading dataset...")
    complete_data = load_dataset("iashchak/ru_common_voice_sova_rudevices_golos_fleurs")
    complete_data = complete_data["train"]

    # FOR TESTING
    complete_data = complete_data[:10]

    audios = complete_data["audio"]
    transcripts = complete_data["text"]

    # workaround for hf datasets library
    complete_data = list(zip(audios, transcripts))
    
    # Set up save directory
    target_path = Path("dataset/target_voice")
    target_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Files will be saved to {target_path.absolute()}")
    
    # Process and save files
    success_count = 0
    index_entries = []

    # Randomly select one sample for target voice
    target_sample = random.choice(complete_data)
    logger.info("Selected target voice sample")
    
    try:
        # Get audio data and transcript
        audio_data = target_sample[0]
        name, signal, sampling_rate = audio_data.values()
        transcript = target_sample[1]
        
        # Generate filename
        filename = name if name else "target_voice"
        filename = filename.replace(".wav", "")  # Remove .wav extension if present
        
        # Save audio file
        if save_audio_file(signal, sampling_rate, str(target_path), filename):
            success_count += 1
            # Add to index entries only if audio save was successful
            index_entries.append((filename, transcript))
            
    except Exception as e:
        logger.error(f"Error processing target voice: {str(e)}")
    
    logger.info(f"Successfully saved target voice to {target_path}")

if __name__ == "__main__":
    main()
