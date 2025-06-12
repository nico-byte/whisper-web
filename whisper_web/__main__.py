import asyncio
import sys

from whisper_web.server import TranscriptionServer


async def main():
    # Start API
    server = TranscriptionServer()
    server.run()

    # Start all tasks in parallel
    await server.execute_event_loop()


if __name__ == "__main__":
    try:
        print("Activating wire...")
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")
