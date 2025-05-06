from datasets import load_dataset
import os
from tqdm import tqdm
import logging
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
    # Load dataset with streaming enabled
    logger.info("Loading dataset in streaming mode...")
    streaming_dataset = load_dataset("iashchak/ru_common_voice_sova_rudevices_golos_fleurs", streaming=True)
    streaming_dataset = streaming_dataset["train"]

    # FOR TESTING
    #streaming_dataset = streaming_dataset.take(10)

    # Set up save directory
    save_path = Path("dataset/source_voices")
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Files will be saved to {save_path.absolute()}")
    
    # Process and save files
    success_count = 0
    index_entries = []
    
    # Using a context manager to create index file incrementally
    index_path = os.path.join(str(save_path), "index.tsv")
    with open(index_path, 'w', encoding='utf-8') as index_file:
        # Process streaming dataset
        for i, entry in enumerate(tqdm(streaming_dataset)):
            try:
                # Get audio data and transcript
                audio_data = entry["audio"]
                name = audio_data.get("path", "")
                signal = audio_data["array"]
                sampling_rate = audio_data["sampling_rate"]
                transcript = entry["text"]

                # Generate filename
                filename = name if name else f"audio_{i}"
                filename = os.path.basename(filename)
                filename = filename.replace(".wav", "")  # Remove .wav extension if present
                
                # Save audio file
                if save_audio_file(signal, sampling_rate, str(save_path), filename):
                    success_count += 1
                    # Write directly to index file
                    index_file.write(f"{filename}\t{transcript}\n")
                    index_entries.append((filename, transcript))  # Keep for count only
                    
            except Exception as e:
                logger.error(f"Error processing file {i}: {str(e)}")
    
    logger.info(f"Successfully saved {success_count} files")
    logger.info(f"Created index.tsv with {len(index_entries)} entries")

if __name__ == "__main__":
    main()
