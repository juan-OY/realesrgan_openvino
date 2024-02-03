#!/usr/bin/env python
# coding: utf-8

#  Run the Real-esrgan with OpenVINO models.
#  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
# https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/202-vision-superresolution/202-vision-superresolution-image.ipynb
# OV model converted by OV2032.2

#Requirement dependency includes 
#pip install "openvino>=2023.1.0"
#pip install opencv-python

import os
import openvino as ov
import time 
import cv2
import numpy as np
from pathlib import Path
#from PIL import Image


class Upscale_rESRGAN(object):
    def __init__(self):
    ##    options=["transcribe", "translate"],

        model_all_local_path = "models/RealESRGAN_x4plus_512.xml"
        model_all_local_path = "./RealESRGAN_x4plus_768x1024.xml"
    
        print("Start Model loading RealESRGAN_x4plus 512x512---")
        self.output_folder = "output/"

        os.makedirs(str(self.output_folder), exist_ok=True)
        t1 = time.time()
        core = ov.Core()
        device_list = core.available_devices
        print("Available device: ", device_list)
        gpu_devices = [device for device in device_list if 'GPU' in device]

        print(gpu_devices)
        if gpu_devices:
            selected_device = gpu_devices[0]  # 选择第一个GPU设备
            print(f"选择的设备是：{selected_device}")
        else:
            selected_device = 'CPU'
            print("没有找到GPU设备，选择CPU。")

        model = core.read_model(model_all_local_path)

        self.input_tensor_name = model.inputs[0].get_any_name()

        input_shape = model.input(self.input_tensor_name).shape

        compiled_model = core.compile_model(model, device_name=selected_device)
        self.output_tensor = compiled_model.outputs[0]

        self.infer_request = compiled_model.create_infer_request()

        self.input_height, self.input_width = list(input_shape)[2:]
        target_height, target_width = list(self.output_tensor.shape)[2:]
        upsample_factor = int(target_height / self.input_height)

        print(f"The network expects inputs with a width of {self.input_width}, " f"height of {self.input_height}")
        print(f"The network returns images with a width of {target_width}, " f"height of {target_height}")
        print("Finish model loading, time : {:.2f} seconds".format(time.time()-t1) )
        return

    #Reference  code https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cf2772fab0af5573da775e7437e6acdca424f26e/modules/esrgan_model.py#L193

    def run(self, image_path):

        t1 = time.time()
        #图像前处理
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise OpenError("Can't open the image from {}".format(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, (self.input_height, self.input_width))
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.reshape(1, *image.shape)   
        #transcription = self.model.transcribe(audio)
        print("image.shape: ", image.shape, self.input_width, self.input_height)
        
        input_data = {self.input_tensor_name: image}
        # Set input tensor for model with one input
        result = self.infer_request.infer(input_data)[self.output_tensor]

        t2 = time.time()
        core = ov.Core()
        upsample_model_path = "RealESRGAN_x4plus_768x1024.xml"
        upsample_model = core.read_model(upsample_model_path)
        compiled_model = core.compile_model(upsample_model, device_name="GPU")
        result = compiled_model(image)[compiled_model.output(0)]

        print("resrgan execution time: {:.2f} seconds".format(t2-t1) )
        output = result[0].squeeze()
        output = np.clip(output, 0, 1) * 255.0
        output = output.astype(np.uint8)
        output = cv2.cvtColor(output.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

        return output

             
if __name__ == "__main__":

    #True menas to use whisper small, other wise use whisper medium
    #image_path = '../inputs/wolf_gray.jpg'
    #image_path = '../inputs/0014.jpg'
    #image_path = '/home/a770/crystal/llm/audiollm/output/2023-09-28_10-08-53.png'
    t0 = time.time()
    image_path = "./7.png"
    output_folder = "output/"
    upscaler = Upscale_rESRGAN()
    image = upscaler.run(image_path)
    #print(image)
    print(image.shape)

    filename = os.path.basename(image_path)  # 获取文件名（例如：'0014.jpg'）
    image_name = os.path.splitext(filename)[0]  # 移除文件扩展名（例如：'0014'）

    output_path = str(Path(output_folder) / (image_name + "x4_s.jpg"))
    print("output path: ", output_path)
    cv2.imwrite(output_path, image)
    print("cost time: ", time.time()-t0)


