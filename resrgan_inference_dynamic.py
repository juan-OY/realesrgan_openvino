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
import math
#from PIL import Image


class Upscale_rESRGAN(object):
    def __init__(self, scale=4, tile=4, tile_pad=10, pre_pad=10,):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
    ##    options=["transcribe", "translate"],

        model_all_local_path = "models/RealESRGAN_x4plus_dynamic.xml"
    
        print("Start Model loading RealESRGAN_x4plus---")
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

        for input_layer in model.inputs:
            print(input_layer.names, input_layer.partial_shape)
        # Oet third and fourth dimensions as dynamic
        model.reshape([1, 3, -1, -1])
        #model.reshape ([1, 3, (512, 768), (512, 768)])

        self.input_tensor_name = model.inputs[0].get_any_name()

        compiled_model = core.compile_model(model, device_name=selected_device)
        self.output_tensor = compiled_model.outputs[0]

        self.infer_request = compiled_model.create_infer_request()

        return

    #Reference code 
    ##https://github.com/xinntao/Real-ESRGAN/blob/5ca1078535923d485892caee7d7804380bfc87fd/realesrgan/utils.py#L88
  
    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = img /255.0
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        print("before image shape: ", img.shape)
        self.img = np.expand_dims(img, axis=0)

        # pre_pad
        print("before pre_pad shape: ", self.img.shape)

    def run(self, image_path):

        t1 = time.time()
        #图像前处理
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise OpenError("Can't open the image from {}".format(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if 1:  # preprocess for tiled image
            self.pre_process(image)
            #self.tile_process()
        else:  ## resize for no tiled image 

            image = cv2.resize(image, (self.input_height, self.input_width))
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.reshape(1, *image.shape)   
        #transcription = self.model.transcribe(audio)
        
        print('input image shape: ', self.img.shape)
        input_data = {self.input_tensor_name: self.img}
        # Set input tensor for model with one input
        result = self.infer_request.infer(input_data)[self.output_tensor]

        t2 = time.time()

        print("resrgan execution time: {:.2f} seconds".format(t2-t1) )
        output = result[0].squeeze()
        output = np.clip(output, 0, 1) * 255.0
        output = output.astype(np.uint8)
        output = cv2.cvtColor(output.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

        return output

             
if __name__ == "__main__":

    #True menas to use whisper small, other wise use whisper medium
    #image_path = '../inputs/wolf_gray.jpg'
    #image_path = '../inputs/children-alpha.png'
    image_path = '/home/a770/crystal/llm/audiollm/output/2023-12-02_14-29-58.png'
    output_folder = "output/"
    upscaler = Upscale_rESRGAN()
    image = upscaler.run(image_path)
    #print(image)
    print(image.shape)

    filename = os.path.basename(image_path)  # 获取文件名（例如：'0014.jpg'）
    image_name = os.path.splitext(filename)[0]  # 移除文件扩展名（例如：'0014'）

    output_path = str(Path(output_folder) / (image_name + "x4.jpg"))
    print("output path: ", output_path)
    cv2.imwrite(output_path, image)


