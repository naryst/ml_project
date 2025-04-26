from datasets import load_dataset
import os
from tqdm import tqdm
import librosa
import logging
from typing import Tuple
from pathlib import Path
import soundfile as sf
import numpy as np
import sys
from tqdm import tqdm

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

def create_index_file(save_path: str, entries: list) -> bool:
    """
    Create index.tsv file with filename and transcript pairs.
    
    Args:
        save_path: Directory where to save index.tsv
        entries: List of (filename, transcript) tuples
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        index_path = os.path.join(save_path, "index.tsv")
        with open(index_path, 'w', encoding='utf-8') as f:
            for filename, transcript in entries:
                f.write(f"{filename}\t{transcript}\n")
        logger.info(f"Successfully created index file at {index_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create index file: {str(e)}")
        return False

def main():
    # Load dataset
    logger.info("Loading dataset...")
    complete_data = load_dataset("iashchak/ru_common_voice_sova_rudevices_golos_fleurs")
    complete_data = complete_data["train"]

    # FOR TESTING
    # complete_data = complete_data[:10]

    audios = complete_data["audio"]
    transcripts = complete_data["text"]

    # workaround for hf datasets library
    complete_data = list(zip(audios, transcripts))

    
    # Set up save directory
    save_path = Path("dataset/data")
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Files will be saved to {save_path.absolute()}")
    
    # Process and save files
    success_count = 0
    index_entries = []

    for i, entry in tqdm(enumerate(complete_data)):
        try:
            # Get audio data and transcript
            audio_data = entry[0]
            name, signal, sampling_rate = audio_data.values()
            transcript = entry[1]

            # Generate filename
            filename = name if name else f"audio_{i}"
            filename = filename.replace(".wav", "")  # Remove .wav extension if present
            
            # Save audio file
            if save_audio_file(signal, sampling_rate, str(save_path), filename):
                success_count += 1
                # Add to index entries only if audio save was successful
                index_entries.append((filename, transcript))
                
        except Exception as e:
            logger.error(f"Error processing file {i}: {str(e)}")
    
    # Create index.tsv file
    if index_entries:
        create_index_file(str(save_path), index_entries)
    
    logger.info(f"Successfully saved {success_count} out of {len(complete_data)} files")
    logger.info(f"Created index.tsv with {len(index_entries)} entries")

if __name__ == "__main__":
    main()
