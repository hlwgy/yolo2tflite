# %%
import numpy as np
import tensorflow as tf
import cv2

def filter_boxes(output, threshold):
    # 获取所有候选框的坐标和得分
    boxes = output[:, :4]
    scores = output[:, 4:]
    # 计算每个边界框最高的得分
    max_scores = np.max(scores, axis=1)
    # 找到满足阈值条件的边界框
    keep = max_scores >= threshold
    # 返回筛选后的边界框和得分
    return boxes[keep], scores[keep]

def iou(box1, box2):
    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # 计算交集区域的面积
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # 计算两个边界框的面积
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    # 计算IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def non_max_suppression(boxes, scores, threshold):
    # 创建一个用于存储保留的边界框的列表
    keep = []
    # 对得分进行排序
    order = scores.argsort()[::-1]
    # 循环直到所有边界框都被检查
    while order.size > 0:
        # 将当前最大得分的边界框添加到keep中
        i = order[0]
        keep.append(i)
        # 计算剩余边界框与当前边界框的IoU
        ious = np.array([iou(boxes[i], boxes[j]) for j in order[1:]])
        # 找到与当前边界框IoU小于阈值的边界框
        inds = np.where(ious <= threshold)[0]
        # 更新order，只保留那些与当前边界框IoU小于阈值的边界框
        order = order[inds + 1]
    return keep

def detect(interpreter, input_image, min_prop=0.90, non_max_value=0.8):

    input_image_f32 = input_image.astype(dtype=np.float32)/ 255
    input_data = np.expand_dims(input_image_f32, axis=0)

    # 获取输入和输出的详细信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # 设置模型的输入
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # 运行模型
    interpreter.invoke()
    # 获取模型输出结果
    detect_scores = interpreter.get_tensor(output_details[0]['index'])
    detect_score = np.squeeze(detect_scores)  # 降维
    output_data = np.transpose(detect_score) # 转换
    # 筛选掉得分低于95%的边界框
    filtered_boxes, filtered_scores = filter_boxes(output_data, min_prop)
    # 计算每个边界框的最高得分
    max_scores = np.max(filtered_scores, axis=1)
    # 应用非极大值抑制，阈值设为0.5
    keep = non_max_suppression(filtered_boxes, max_scores, non_max_value)
    # 最后留下的候选框
    final_boxes = filtered_boxes[keep]
    final_scores = filtered_scores[keep]
    indexs = np.argmax(final_scores, axis=1)
    return final_boxes, indexs

def pre_img(image):
    height, width, _ = image.shape

    # 等比例缩放
    if height > width:
        new_height = 640
        new_width = int(640 * width / height)
    else:
        new_width = 640
        new_height = int(640 * height / width)
    image_resized = cv2.resize(image, (new_width, new_height))

    # 创建一个640*640的白色背景图像
    background = np.ones((640, 640, 3), dtype=np.uint8) * 255

    # 将缩放后的图像粘贴到背景图像的中心位置
    start_x = (640 - new_width) // 2
    start_y = (640 - new_height) // 2
    background[start_y:start_y+new_height, start_x:start_x+new_width] = image_resized
    return background

# %%
# 加载TFLite模型并获取interpreter
interpreter = tf.lite.Interpreter(model_path='best_num_int8.tflite')
interpreter.allocate_tensors()

# 假设你有一个输入图像（预处理过）
input_image = cv2.imread('num.jpg')
# 调整图像大小
input_image = pre_img(input_image)
final_boxes, indexs = detect(interpreter, input_image, min_prop=0.80, non_max_value=0.9)
names = ["num_1", "num_2"]

print("识别到", [names[i] for i in indexs])

colors = [(0, 0, 255), (255, 0, 0)]
width, height = 640, 640

for i, dt in enumerate(final_boxes):
    center_x = int(dt[0]*width)
    center_y = int(dt[1]*height)
    w = int(dt[2] * width)
    h = int(dt[3] * height)
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)
    index = indexs[i]
    # 使用 cv2.putText() 方法绘制文字
    cv2.rectangle(input_image, (x,y), (x+w, y+h), colors[index], 2)
    cv2.putText(input_image, names[index], (x,y-4), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[index], 2, cv2.LINE_AA)

# 显示图像
cv2.imshow("Image with Rectangle", input_image)
cv2.waitKey(0) 
cv2.destroyAllWindows()
# %%
