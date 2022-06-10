# August
# 开发时间：9/6/2022 下午1:36
from getdata import prepaer_dataset
import tensorflow as tf
from TFrecord import read_TFRecord_file

#定义VGG-16模型
def vgg16_model(input_shape=(224,224,3)):
    vgg16=tf.keras.applications.vgg16.VGG16(include_top=False,
                                           weights='imagenet',
                                           input_shape=input_shape)
    for layer in vgg16.layers:
        layer.trainable=False

    last=vgg16.output
    #加入剩下未经训练的全连接层
    x=tf.keras.layers.Flatten()(last)
    x=tf.keras.layers.Dense(128,activation='relu')(x)
    x=tf.keras.layers.Dropout(0.3)(x)
    x=tf.keras.layers.Dense(32,activation='relu')(x)
    x=tf.keras.layers.Dropout(0.3)(x)
    x=tf.keras.layers.Dense(2,activation='softmax')(x)

    model=tf.keras.models.Model(inputs=vgg16.input,outputs=x)
    model.summary()
    return model

if __name__=='__main__':
    #新建模型
    model=vgg16_model()
    #模型设置
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #读取训练数据
    buffer_size=2000#定义缓冲器大小和批次
    batch_size=8
    train_data_dir='./data/train_small/'
    #dataset_train=prepaer_dataset(train_data_dir,buffer_size,batch_size)

    tfrecord_file=train_data_dir+'train.tfrecords'
    dataset_train=read_TFRecord_file(tfrecord_file)
    dataset_train=dataset_train.shuffle(buffer_size)
    dataset_train=dataset_train.batch(batch_size)

    training_epochs=3
    #进行训练
    train_history=model.fit(dataset_train,epochs=training_epochs,verbose=1)
    #模型存储
    yaml_string=model.to_yaml()
    with open('./models/cat_dog.yaml','w')as model_file:#模型结构存储
        model_file.write(yaml_string)
    model.save_weights('./models/cat_dog.h5')#模型权重参数存储

