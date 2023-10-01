# The Quadric Version of onnxruntime

This repository contains the a distribution of onnxruntime with additional operator quantization capabilities.


## Prerequisites:
- python 3.9
- pip

## Clone repository and build:
```
git clone --recursive https://github.com/quadric-io/onnxruntime onnxruntime
cd onnxruntime
python3.9 -m venv venv
source venv/bin/activate
# Install required packages. numpy version is restricted by TVM
pip3 install wheel packaging numpy==1.24.4
# Build the python package
./build.sh --build_wheel --config Release --parallel
```

## Install 
```
# Find the wheel you just created
$ find . -name '*.whl'
./build/MacOS/Release/dist/onnxruntime-1.16.0-cp39-cp39-macosx_13_0_arm64.whl
# Install it
pip3 install ./build/MacOS/Release/dist/onnxruntime-1.16.0-cp39-cp39-macosx_13_0_arm64.whl
```
