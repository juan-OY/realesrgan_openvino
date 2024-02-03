import openvino as ov

onnx_path = 'RealESRGAN_x4plus_512.onnx'
onnx_path = 'RealESRGAN_x4plus_1024x768.onnx'
#ov_model = ov.convert_model('RealESRGAN_x4plus_512.onnx')
ov_model = ov.convert_model(onnx_path)

#ov.save_model(ov_model, 'RealESRGAN_x4plus_512.xml')
ov.save_model(ov_model, onnx_path.split(".")[0]+ '.xml')
