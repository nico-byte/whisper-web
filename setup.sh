echo "Setting up Whisper Web environment..."
echo "Copying .env.example to .env..."
cp .env.example .env

echo "Creating .models directory..."
mkdir -p .models
chmod 755 .models

echo "Creating Docker volume for models..."
docker volume create --driver local -o o=bind -o type=none -o device="${PWD}/.models" models