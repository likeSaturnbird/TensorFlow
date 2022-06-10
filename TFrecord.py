# August
# 开发时间：10/6/2022 下午1:48
import tensorflow as tf
import os
import matplotlib.pyplot as plt
def read_image_filenames(data_dir):
    #说明：读取指定数据目录下的图片文件信息，返回文件名列表和标签列表(没有采用tf.constant方式)
    cat_dir=data_dir+'cat/'
    dog_dir=data_dir+'dog/'

    cat_filenames = [cat_dir+fn for fn in os.listdir(cat_dir)]
    dog_filenames = [dog_dir + fn for fn in os.listdir(dog_dir)]

    filename=cat_filenames+dog_filenames
    labels=[0]*len(cat_filenames)+[1]*len(dog_filenames)
    return filename,labels

def write_TFRecord_file(filenames,labels,tfrecord_file):
    #说明：生成TFRecord格式数据文件函数
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for filename,label in zip(filenames,labels):
            #读取数据集图片到内存，image为一个Byte类型的字符串
            image=open(filename,'rb').read()
            #建立tf.train.Feature字典
            feature={
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            #通过feature字典建立Example
            example=tf.train.Example(features=tf.train.Features(feature=feature))
            #将Example序列化并写入TFRecord文件
            writer.write(example.SerializeToString())
#定义一个Feature结构，告诉解码器每个Feature的类型是什么
feature_description={
    'image':tf.io.FixedLenFeature([],tf.string),
    'label':tf.io.FixedLenFeature([],tf.int64),
    }

def parse_example(example_string):
    #说明：将tf.train.example解码
    feature_dict=tf.io.parse_single_example(example_string,feature_description)
    feature_dict['image']=tf.io.decode_jpeg(feature_dict['image']) #解码JPEG图片
    feature_dict['image']=tf.image.resize(feature_dict['image'],[224,224])/255.0
    return feature_dict['image'],feature_dict['label']

def read_TFRecord_file(tfrecord_file):
    #说明:读取TFRecord文件
    raw_dataset=tf.data.TFRecordDataset(tfrecord_file)
    #解码
    dataset=raw_dataset.map(parse_example)
    return dataset
if __name__=='__main__':
    train_data_dir='./data/train_small/'
    tfrecord_file=train_data_dir+'train.tfrecords'
    if not os.path.isfile(tfrecord_file):#train_data_dir
        train_filenames,train_labels=read_image_filenames(train_data_dir)
        write_TFRecord_file(train_filenames,train_labels,tfrecord_file)
        print('生成TFRecord文件:',tfrecord_file)
    else:
        print(tfrecord_file,'该文件已存在')

    #Dataset的数据缓冲器大小,和数据集大小及规律有关
    buffer_size=2000
    #Dataset的数据批次大小，每批次多少个样本数
    batch_size=8
    dataset_train=read_TFRecord_file(tfrecord_file)
    dataset_train=dataset_train.shuffle(buffer_size)
    dataset_train=dataset_train.batch(batch_size)

#####Test
    sub_dataset=dataset_train.take(1)
    for images,labels in sub_dataset:
        fig,axs=plt.subplots(1,batch_size)
        for i in range(batch_size):
            axs[i].set_title(labels.numpy()[i])
            axs[i].imshow(images.numpy()[i])
            axs[i].set_xticks([]);axs[i].set_yticks([]);
        plt.show()