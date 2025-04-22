from datasets import load_dataset
import os
from tqdm import tqdm
import librosa
import logging
from typing import Tuple
from pathlib import Path
import soundfile as sf
import numpy as np

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
    
    # Set up save directory
    save_path = Path("dataset/data")
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Files will be saved to {save_path.absolute()}")
    
    # Process and save files
    success_count = 0
    for i in tqdm(range(len(complete_data)), desc="Saving audio files"):
        try:
            name, signal, sampling_rate = complete_data[i]["audio"].values()
            filename = name if name else f"audio_{i}"
            if save_audio_file(signal, sampling_rate, str(save_path), filename):
                success_count += 1
        except Exception as e:
            logger.error(f"Error processing file {i}: {str(e)}")
    
    logger.info(f"Successfully saved {success_count} out of {len(complete_data)} files")

if __name__ == "__main__":
    main()
