# realesrgan_openvino
Image upscaling with Real-ESRGAN with OpenVINO acceleration on intel platform

## Model conversion 
Use Real-ESRGAN to convert the model from pytorch to onnx
set up the Real-ESRGAN environment 
```
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop
```
Download the weight file 
```
Wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```
### Convert model from pytorch to onnx with static shape
Real-ESRGAN provides one pytoch to onnx scripts under: scripts/pytorch2onnx.py, to export the onnx model as static input, please below x Value heigh and width to some specific values as below.
```
x = torch.rand(1, 3, 512, 512)

onnx_path = "RealESRGAN_x4plus_512.onnx"
torch.onnx.export(model,               # model being run
                x,                         # model input (or a tuple for multiple inputs)
                onnx_path,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=12,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output']) # the model's output names
)
```
### Convert model from pytorch to onnx with static shape
PyTorch dynamic axes empower models to adjust to input data with diverse dimensions. In the example below, the height and width are set as dynamic. Please refer to the provided code for exporting. If the upscaling model is required to handle images with varying shapes, you can employ this approach. However, it comes with a trade-off, as it may result in increased inference time.
```
dynamic_axes = {
        'input': { 2: 'height', 3: 'width'}
}
onnx_path = "RealESRGAN_x4plus_dynamic.onnx"
torch.onnx.export(model,               # model being run
                x,                         # model input (or a tuple for multiple inputs)
                onnx_path,                # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=12,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output']   # the model's output names
                dynamic_axes = dynamic_axes  # Juan add, save the model as dynamic shape
                )    

```
## Install the required dependencise 
```
pip install "openvino>=2023.1.0"
pip install opencv-python
```
## Convert Model to MO file
```
python model_convert.py
```

## run the model
```
python resrgan_inference_static.py
```
