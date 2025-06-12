import torch


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
