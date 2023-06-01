# The Quadric Version of onnxruntime

This repository contains the a distribution of onnxruntime with additional operator quantization capabilities.


## Prerequisites:
- python 3.9
- pip

## Clone repository and build:
```
git clone --recursive https://github.com/quadric-io/onnxruntime onnxruntime
cd onnxruntime
# Install wheel
pip3 install wheel
# Build the python package
./build.sh --build_wheel --config Release --parallel --compile_no_warning_as_error --skip_tests --skip_submodule_sync
```

## Install 
```
# Find the wheel you just created
$ find . -name '*.whl'
./build/MacOS/Release/dist/onnxruntime-1.15.1-cp310-cp310-macosx_13_0_arm64.whl
# Install it
pip3 install ./build/MacOS/Release/dist/onnxruntime-1.15.1-cp310-cp310-macosx_13_0_arm64.whl
```
