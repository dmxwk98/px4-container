from random import *
#import airsim
# from tensorflow.keras.models import load_model
import onnx
import onnxruntime as ort
import cv2
import numpy as np
# import torch


class JBNU_Collision():
    def __init__(self):
        print("initialJBNU")
        
    def CA(self, img2d):
        """
        responses = self.client.simGetImages([airsim.ImageRequest("", airsim.ImageType.DepthPerspective,  True,False)]) 
        response = responses[0]
        if ((responses[0].width != 0 or responses[0].height != 0)):
            img1d = response.image_data_float
            img2d = np.reshape(img1d,(response.height, response.width)) 
        else:
            print("Something bad happened! Restting AirSim!")
            img2d = np.ones(640, 480)
        """
        Image = img2d
        
        Image = cv2.resize(Image, (200,200), cv2.INTER_AREA)
        Image = np.expand_dims(Image, axis=0)
        
        onnx_model = onnx.load("/root/ros_ws/src/a4vai/a4vai/collision_avoidance/feedforward.onnx")
        onnx.checker.check_model(onnx_model)
        ort_session = ort.InferenceSession("/root/ros_ws/src/a4vai/a4vai/collision_avoidance/feedforward.onnx")
        # print(onnx.helper.printable_graph(onnx_model.graph))
        input_name = ort_session.get_inputs()[0].name
        Act = ort_session.run(None, {input_name:Image.astype(np.float32)})
        vx = Act[0][0][0]
        vy = 0.0
        vz = Act[0][0][1]
        vyaw = -Act[0][0][2]
        print(type(vx))
        return float(vx),float(vy),float(vz), -float(vyaw)

        

# if __name__ == '__main__':
#     tensor = JBNU_Collision()
#     vx,vy,vz,vyaw =tensor.main()
#     print(vx,vy,vz,vyaw)
 
