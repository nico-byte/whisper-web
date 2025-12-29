# This script fixes the rpath of the torchcodec shared libraries to include the ffmpeg library path on macOS (Homebrew installation).
# This is necessary because the torchcodec package doesn't find the ffmpeg libraries by default.
install_name_tool -add_rpath /opt/homebrew/Cellar/ffmpeg/lib .venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_core7.dylib
install_name_tool -add_rpath /opt/homebrew/Cellar/ffmpeg/lib .venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_custom_ops7.dylib
install_name_tool -add_rpath /opt/homebrew/Cellar/ffmpeg/lib .venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_pybind_ops7.so

codesign --force -s - .venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_core7.dylib
codesign --force -s - .venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_custom_ops7.dylib
codesign --force -s - .venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_pybind_ops7.so