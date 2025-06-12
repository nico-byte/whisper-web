import os


def is_running_in_docker() -> bool:
    """
    Detect if the application is running inside a Docker container.

    Returns:
        bool: True if running in Docker, False otherwise
    """
    # Check for common Docker indicators
    docker_indicators = [
        # Check for .dockerenv file (most reliable)
        os.path.exists("/.dockerenv"),
        # Check if running as PID 1 (common in containers)
        os.getpid() == 1,
        # Check for Docker-specific environment variables
        os.environ.get("DOCKER_CONTAINER") is not None,
        # Check for container-specific cgroup info
        _is_in_container_cgroup(),
    ]

    return any(docker_indicators)


def _is_in_container_cgroup() -> bool:
    """Check if running in a container based on cgroup information."""
    try:
        with open("/proc/1/cgroup", "r") as f:
            cgroup_content = f.read()
            return "docker" in cgroup_content or "containerd" in cgroup_content
    except (FileNotFoundError, PermissionError):
        return False


def get_server_urls():
    """Get server URLs based on environment."""
    if not is_running_in_docker():
        return "http://127.0.0.1:8000", "ws://127.0.0.1:8000"

    server_type = os.getenv("SERVER_TYPE", "cpu")
    service_name = f"server-{server_type}"
    return f"http://{service_name}:8000", f"ws://{service_name}:8000"
