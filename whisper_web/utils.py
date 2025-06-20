import os
import torch
import re


def set_device(device) -> torch.device:
    # Order of preference for fallback
    fallback_order = ["cuda", "mps", "cpu"]

    print(f"Requested device: {device}")

    def test_device(device_str: str) -> bool:
        """Test if a device is available and functional."""
        try:
            test_device = torch.device(device_str)
            # Test tensor creation and basic operation
            test_tensor = torch.tensor([[0, 3], [5, 7]], dtype=torch.float32, device=test_device)
            # Simple operation to ensure device works
            _ = test_tensor + 1
            return True
        except Exception as e:
            print(f"Device {device_str} test failed: {e}")
            return False

    # First try the requested device
    if device in fallback_order and test_device(device):
        print(f"Using requested device: {device}")
        return torch.device(device)

    # If requested device failed, try alternatives
    if device in fallback_order:
        print(f"Requested device '{device}' not available, trying alternatives...")
        fallback_order.remove(device)

    # Try devices in fallback order
    for fallback_device in fallback_order:
        if fallback_device != device and test_device(fallback_device):
            print(f"Using fallback device: {fallback_device}")
            return torch.device(fallback_device)

    # Ultimate fallback to CPU (should always work)
    print("All devices failed, forcing CPU")
    return torch.device("cpu")


def get_installed_models():
    """Scan the .models folder and return formatted model names."""
    models_path = "./.models"  # Use the current working directory
    available_models = []

    # Ensure the models path exists
    if not os.path.exists(models_path):
        print(f"Models folder does not exist: {models_path}")
        return []

    try:
        if os.path.exists(models_path):
            # List all directories in .models folder
            for item in os.listdir(models_path):
                item_path = os.path.join(models_path, item)
                # Check if it's a directory and starts with "models--"
                if os.path.isdir(item_path) and item.startswith("models--"):
                    # Format: models--distil-whisper--distil-large-v3 -> distil-whisper/distil-large-v3
                    parts = item.split("--")
                    if len(parts) >= 3:
                        # Skip the "models" part and join the rest with "/"
                        formatted_name = "/".join(parts[1:])
                        available_models.append(formatted_name)

        # Sort models for consistent ordering
        available_models.sort()

        # Add fallback models if no models found
        if not available_models:
            return []

    except Exception as e:
        print(f"Error scanning models folder: {e}")
        # Fallback models
        return []

    return available_models


def process_transcription_timestamps(transcriptions: list[str], last_timestamp: float) -> tuple[list[str], float]:
    """Process transcriptions to maintain timestamp continuity across batches.

    :param transcriptions: List of transcription strings with timestamps
    :return: List of transcriptions with adjusted timestamps
    """
    processed_transcriptions = []

    for transcription in transcriptions:
        # Extract all timestamps from the transcription
        timestamp_pattern = r"<\|(\d+\.\d+)\|>"
        timestamps = re.findall(timestamp_pattern, transcription)

        if not timestamps:
            processed_transcriptions.append(transcription)
            continue

        # Convert to float and find the highest timestamp in this transcription
        float_timestamps = [float(ts) for ts in timestamps]
        max_timestamp = max(float_timestamps)

        # Adjust all timestamps by adding the last_timestamp offset
        adjusted_transcription = transcription
        for ts in timestamps:
            old_timestamp = f"<|{ts}|>"
            new_timestamp_value = float(ts) + last_timestamp
            new_timestamp = f"<|{new_timestamp_value:.2f}|>"
            adjusted_transcription = adjusted_transcription.replace(old_timestamp, new_timestamp, 1)

        processed_transcriptions.append(adjusted_transcription)

        # Update last_timestamp with the highest timestamp from this batch
        last_timestamp += max_timestamp

    return processed_transcriptions, last_timestamp
