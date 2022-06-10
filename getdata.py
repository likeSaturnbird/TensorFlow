# August
# 开发时间：9/6/2022 上午9:22
###导入库函数
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

# region 显示TensorFlow及Keras版本
# # print("TensorFlow version is",tf.__version__)
# # print("Keras version is",tf.keras.__version__)
# # print("GPU is","available"if tf.test.is_gpu_available() else "Not available")
# #
# # ###测试例子
# # x=tf.constant([10001,10002,10003,10004,10005]);y=tf.constant([1,2,3,4,5])
# # dataset=tf.data.Dataset.from_tensor_slices((x,y))
# # for i,j in dataset:
# #     print(i.numpy(),j.numpy())
# # #使用迭代器
# # x=np.array([10001,10002,10003,10004,10005]);y=np.array([1,2,3,4,5])
# # dataset=tf.data.Dataset.from_tensor_slices((x,y))
# # it=iter(dataset)
# # i,j=it.next();print(i.numpy(),j.numpy())
# # i,j=it.next();print(i.numpy(),j.numpy())
# endregion
###构建数据集
def read_image_filenames(data_dir):
    #说明：读取指定数据目录下的图片文件信息，返回文件名列表和标签列表
    cat_dir=data_dir+'cat/'
    dog_dir=data_dir+'dog/'

    cat_filenames = tf.constant([cat_dir+fn for fn in os.listdir(cat_dir)])
    dog_filenames = tf.constant([dog_dir + fn for fn in os.listdir(dog_dir)])
    filename=tf.concat([cat_filenames,dog_filenames],axis=-1)
    labels=tf.concat([
        tf.zeros(cat_filenames.shape,dtype=tf.int32),
        tf.ones(dog_filenames.shape,dtype=tf.int32)
    ],axis=-1)
    return filename,labels

def decode_image_and_resize(filename,lable):
    #说明:读取图片文件并解码,调整图片的大小并标准化
    image_string=tf.io.read_file(filename)
    image_decoded=tf.image.decode_jpeg(image_string)
    image_resized=tf.image.resize(image_decoded,[224,224])/255.0
    return image_resized,lable
# region

# train_data_dir='./data/train/'
# filenames,labels=read_image_filenames(train_data_dir)
# dataset=tf.data.Dataset.from_tensor_slices((filenames,labels))
# #test(取前几个数据)
# sub_dataset=dataset.take(3)
# for x,y in sub_dataset:
#     print('filename:',x.numpy(),'label:',y.numpy())
# #读取图像数据并处理
# dataset=dataset.map(
#     map_func=decode_image_and_resize,num_parallel_calls=tf.data.experimental.AUTOTUNE
# )
# #test2(查看前几个样本)
# sub_dataset=dataset.take(3)
# for x,y in sub_dataset:
#     print(x.shape,y.shape)
# #######打乱数据集
# buffer_size=100 #缓冲区
# dataset=dataset.shuffle(buffer_size)
# sub_dataset=dataset.take(3)
# for x, y in sub_dataset:
#     plt.title(y.numpy())
#     plt.imshow(x.numpy())
#     plt.show()
# batch_size=8
# dataset=dataset.batch(batch_size)
# sub_dataset=dataset.take(3)
# for x, y in sub_dataset:
#     print(x.shape, y.shape)
# endregion
def prepaer_dataset(data_dir,buffer_size,batch_size):
    filenames,labels=read_image_filenames(data_dir)
    dataset=tf.data.Dataset.from_tensor_slices((filenames,labels))
    dataset = dataset.map(
        map_func=decode_image_and_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset=dataset.shuffle(buffer_size)
    dataset=dataset.batch(batch_size)
    dataset=dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
if __name__=='__main__':
    buffer_size=2000#定义缓冲器大小和批次
    batch_size=8
    train_data_dir='./data/train_small/'
    dataset_train=prepaer_dataset(train_data_dir,buffer_size,batch_size)
    sub_dataset=dataset_train.take(1)
    for images,labels in sub_dataset:
        fig,axs=plt.subplots(1,batch_size)
        for i in range(batch_size):
            axs[i].set_title(labels.numpy()[i])
            axs[i].imshow(images.numpy()[i])
            axs[i].set_xticks([]);axs[i].set_yticks([]);
        plt.show()