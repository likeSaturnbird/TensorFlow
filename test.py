# August
# 开发时间：9/6/2022 下午3:19
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
def read_image_files(path,start,finish,image_size=(224,224)):
    #说明:path测试集路径；start起始位置；finsh结束位置
    test_files=os.listdir(path)
    test_images=[]
    #读取测试图片进行预处理
    for fn in test_files[start:finish]:
        img_filename=path+fn
        img = image.load_img(img_filename,target_size=image_size)
        img_array=image.img_to_array(img)
        test_images.append(img_array)
    test_data=np.array(test_images)
    test_data /=255.0
    print('你选的图片从第 %d 张到 第%d张'%(start,finish))
    print('测试数据大小为',end='');print(test_data.shape)
    return test_data

def test_image_predict(path,start,finish,image_size=(224,224)):
    # 读取测试图片并预处理
    test_data=read_image_files(path,start,finish,image_size=(224,224))
    #执行预测
    preds=model.predict(test_data)
    #显示图片及预测结果
    for i in range(0,finish-start):
        if np.argmax(preds[i])==0:
            label='cat'+str(preds[i][0])
        else:
            label = 'dog' + str(preds[i][1])
        plt.title(label)
        plt.imshow(test_data[i])
        plt.axis('off')
        plt.show()
if __name__=='__main__':
    #恢复模型的结构
    with open('./models/cat_dog.yaml') as yamlfile:
         loaded_model_yaml=yamlfile.read()
    model=tf.keras.models.model_from_yaml(loaded_model_yaml)
    #导入模型的权重参数
    model.load_weights('./models/cat_dog.h5')
    #执行预测
    test_data_dir='./data/test1/'
    test_image_predict(test_data_dir,0,5)
