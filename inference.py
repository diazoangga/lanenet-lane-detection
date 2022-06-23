import cv2
import numpy as np

import onnxruntime as ort

def preprocessing_img(img_path):
    img = cv2.resize(img_path, [512,256])
    img = np.array(img).astype(np.float32)/255.0
    img = np.expand_dims(img, axis=0)
    return img

def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def find_center(bin_pred):
    if bin_pred.shape == [256,512]:
        center = -1
    else:
        ocv = bin_pred[220, :]
        ocv = cv2.blur(ocv,(9,9))
        _,ocv = cv2.threshold(ocv, 190, 255, 0)
        # collapse to line
        ocv = cv2.ximgproc.thinning(ocv,0)
        c, _ = cv2.findContours(ocv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        points = []
        if len(c) > 1:
            for i in c:
                points = np.append(points, i[0,0,1])
                # print(i[0,0,1])
                
            points = points - 256
            left, right = np.argsort(abs(points))[0:2]
            center = (points[left] + points[right])/2 + 256
        else:
            center = -1
    return center

def inference_bisenetv2(imgRGB, bin_pred, inst_pred):
    temp = inst_pred
    for i in range(3):
        temp[:,:,i] = minmax_scale(np.array(inst_pred[:, :, i]))
        embedding_image = np.array(temp, np.uint8)
    bin_pred_unnormalized=(np.argmax(bin_pred, axis=-1)*255).astype(np.uint8)
    bin_pred_img = np.expand_dims(bin_pred_unnormalized, axis=-1)
    bin_pred_img = np.concatenate((bin_pred_img,bin_pred_img,bin_pred_img),axis=-1)
    bin_pred_img = (bin_pred_img).astype(np.uint8)
    # re_a = (re_a*inst_pred[:,:,1:4]).astype(np.uint8)*255
    #k = minmax_bin(np.array(re_a))
    #h = (k > 0.3).astype(np.uint8)
    lane_img = embedding_image[:,:,(2,1,0)]*bin_pred_img
    input_img = (imgRGB[0,:,:,:]*255).astype(np.uint8)
    # print(input_img.shape, y.shape)
    out_img = cv2.addWeighted(input_img, 0.5, lane_img, 1, 0.0)

    center_lane = find_center(bin_pred_unnormalized)
    if center_lane != -1:
        out_img = cv2.line(out_img, [int(center_lane),256], [int(center_lane),150], (255,255,255), 1)

    # dst, center = centerLine(a, dst)
    return out_img, center_lane

weights_path = "./carla_weight_onnx/model.onnx"
image_path = './data/testing_data_example/gt_image/0.png'
print(f'Built model with weights {weights_path}...')
model = ort.InferenceSession(weights_path, providers=["CUDAExecutionProvider"])

input_image = preprocessing_img(0, image_path)
result_model = model.run(["bise_net_v2_1", "bise_net_v2_1_1"], {"input_1": input_image})
bin_pred = np.squeeze(result_model[0], axis=0)
inst = np.squeeze(result_model[1], axis=0)

out_img, center_lane = inference_bisenetv2(input_image, bin_pred, inst)

cv2.imshow('lane detection', out_img)
cv2.waitKey(0)