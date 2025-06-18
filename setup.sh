echo "Setting up Whisper Web environment..."
echo "Copying .env.example to .env..."
cp .env.example .env

echo "Creating .models directory..."
mkdir -p .models
chmod 755 .models