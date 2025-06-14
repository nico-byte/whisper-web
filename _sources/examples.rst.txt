Examples
=============

Installation/Usage:
*******************
We recommend using ``uv`` as the python package manager.
How to install uv can be found at https://docs.astral.sh/uv/getting-started/installation/.
To install the dependencies, run the following command:
.. code-block::
    $ uv sync

After completing the installation, one can now use the transcriber.

Simply start the server and the cli client:
.. code-block::
    # In two seperate terminal sessions
    $ uv run python -m server
    $ uv run python -m cli_client

One can also run a frontend app. We provide a simple streamlit app that shows the transcriptions. The CLI Client still has to be running in the background.
.. code-block::
    # In another terminal session
    $ uv run python -m streamlit run streamlit_app

We also provide a Makefile to make it easier to run the examples. You can run the following command to run the server, cli client and streamlit app all at once:
.. code-block::
    $ make run

Low Level Usage:
****************
Running the server in a python script:

.. code-block:: python

    import asyncio
    import sys

    from whisper_rt.whisper_model import ModelConfig
    from whisper_rt.server import TranscriptionServer


    async def main():
        model_config = ModelConfig()

        # Start API
        server = TranscriptionServer(model_config=model_config)
        server.run()

        # Start all tasks in parallel
        await server.execute_event_loop()


    if __name__ == "__main__":
        try:
            print("Activating wire...")
            asyncio.run(main())
        except KeyboardInterrupt:
            sys.exit("\nInterrupted by user")

.. note:: 
    The server will receive the audio data from the client and manages the transcription process.

Running a simple client in a python script:

.. code-block:: python
    
    import asyncio
    import websockets
    import soundfile as sf
    import io
    import requests

    from whisper_rt import GeneratorConfig, InputStreamGenerator, GeneratorManager

    BASE_ADDRESS = "127.0.0.1:8000"
    API_BASE_URL = f"http://{BASE_ADDRESS}"
    WS_BASE_URL = f"ws://{BASE_ADDRESS}/ws/transcribe"

    async def stream_audio(generator_manager):
        async with websockets.connect(WS_BASE_URL) as ws:
            print("Connected to WebSocket")
            generator_manager.websocket_status.set()

            try:
                while True:
                    if generator_manager.generator_status.is_set():

                        audio_np = generator_manager.audio  # Should return np.ndarray (shape: [samples, channels])

                        # Convert to WAV bytes in-memory
                        with io.BytesIO() as buffer:
                            sf.write(buffer, audio_np, samplerate=16000, format='WAV')  # or whatever your sample rate is
                            wav_bytes = buffer.getvalue()

                        await ws.send(wav_bytes)
                        print("Sent audio chunk")
                        generator_manager.generator_status.clear()
                        generator_manager.websocket_status.set()
                    
                    await asyncio.sleep(0.25)  # Prevent busy waiting

                    current = requests.get(f"{API_BASE_URL}/transcription/current")
                    final = requests.get(f"{API_BASE_URL}/transcription/final")
                
                    current_text = current.json().get("current_transcription", "[No data]")
                    final_text = final.json().get("current_transcription", "[No data]")
                
                    print(f"Current: {current_text}")
                    print(f"Final: {final_text}")

            except Exception as e:
                print("Error during streaming:", e)

            await ws.close()
        

    async def main():
        generator_config = GeneratorConfig(from_file="./jfk_speech_fub.mp3")
        generator_manager = GeneratorManager()
        generator = InputStreamGenerator(generator_config, generator_manager)
    
        inputstream_task = asyncio.create_task(generator.process_audio())
        stream_task = asyncio.create_task(stream_audio(generator_manager))
        await asyncio.gather(inputstream_task, stream_task)


    if __name__ == "__main__":
        asyncio.run(main())

.. note::
    The client will read the audio data from the input stream and send it to the server for transcription.
    The input stream consists of a custom python module :class:`InputStreamGenerator` that reads audio data from a file or a microphone.