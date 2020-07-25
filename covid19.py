import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate,GlobalMaxPooling2D
from tensorflow.keras.metrics import CategoricalAccuracy, Recall, Precision

def load_image(img_path,size=(80,80)):
    if tf.strings.regex_full_match(img_path, ".*COVID19.*"):
        label = tf.constant(0, tf.uint8)
    elif tf.strings.regex_full_match(img_path, ".*NORMAL.*"):
        label = tf.constant(1, tf.uint8)
    else:
        label = tf.constant(2, tf.uint8)
    label = tf.one_hot(label,depth=3)

    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img,channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    if tf.strings.regex_full_match(img_path, ".*train.*"):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img,0.1)
        img = tf.image.random_contrast(img,0.1,0.2)
        img = tf.image.random_saturation(img,0,5)
        img = tf.image.resize(img,size=(96,96))
        img = tf.image.random_crop(img,size=[80,80,3])

    img = tf.image.resize(img, size)
    return (img, label)

def scheduler(epoch, lr=0.0001):
    if epoch < 3:
        lr=0.0004
    elif epoch <6:
        lr=0.0002
    else:
        lr=0.0001

    return lr

def myModel():
    inputs = Input(shape=(80,80,3))
    conv1 = Conv2D(32, (3, 3), padding='same', activation='elu')(inputs)
    conv1 = Conv2D(32, (3, 3), padding='same', activation='elu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', activation='elu')(pool1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='elu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3),padding='same', activation='elu')(pool2)
    conv3 = Conv2D(128, (3, 3),padding='same', activation='elu')(conv3)
    conv3 = Conv2D(128, (3, 3),padding='same', activation='elu')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)


    conv4 = Conv2D(128, (3, 3),padding='same', activation='elu')(pool3)
    conv4 = Conv2D(128, (3, 3),padding='same', activation='elu')(conv4)
    conv4 = Conv2D(128, (3, 3),padding='same', activation='elu')(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    concat = Concatenate()(
        [GlobalMaxPooling2D()(pool1),
         GlobalMaxPooling2D()(pool2),
         GlobalMaxPooling2D()(pool3),
         GlobalMaxPooling2D()(pool4)])

    flatten1 = Flatten()(concat)
    flatten2 = Flatten()(conv4)
    concat2 = Concatenate()([flatten1,flatten2])

    x = Dense(4096, activation='elu')(concat2) 
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='elu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(3, activation='softmax')(x)

    return models.Model(inputs=inputs,outputs=outputs)
    
if __name__ == "__main__":

    BATCH_SIZE = 32
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="D:\Python\logs")

    # 使用并行化预处理num_parallel_calls 和预存数据prefetch来提升性能
    train_data = tf.data.Dataset.list_files("D:\Python\covid19\chest_xray_dataset/train/*/*.*")\
                                .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                .shuffle(buffer_size=1280).batch(BATCH_SIZE) \
                                .prefetch(tf.data.experimental.AUTOTUNE)
    test_data = tf.data.Dataset.list_files("D:\Python\covid19\chest_xray_dataset/test/*/*.*")\
                                .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                .batch(BATCH_SIZE) \
                                .prefetch(tf.data.experimental.AUTOTUNE)
    covid_data = tf.data.Dataset.list_files("D:\Python\covid19\chest_xray_dataset/test/COVID19/*.*")\
                                .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                .batch(100) \
                                .prefetch(tf.data.experimental.AUTOTUNE)
                                   
    model = myModel()

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.12)
    adam = tf.keras.optimizers.Adam(learning_rate=0.0004)
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model.compile(
                loss=loss,
                optimizer=adam,
                metrics=[CategoricalAccuracy(name="accuracy"),Recall(),Precision()]
                  )

    model.fit(train_data,
              epochs=12,
              class_weight={0:10,1:3,2:1},
              validation_data=test_data,
              callbacks=[tensorboard_callback,scheduler_callback]
              )

    model.evaluate(test_data)
    model.evaluate(covid_data)
    model.save('covid19.h5')

'''

After 10 epoches: 

Train result
# 88/88 [==============================] - 267s 3s/step - loss: 0.9860 - accuracy: 0.9325 - recall: 0.9133 - precision: 0.9463 - val_loss: 
# 0.5632 - val_accuracy: 0.8730 - val_recall: 0.8446 - val_precision: 0.8967

Test result
# 12/12 [==============================] - 9s 768ms/step - loss: 0.5632 - accuracy: 0.8730 - recall: 0.8446 - precision: 0.8967

COVID-19 result
# 1/1 [==============================] - 0s 999us/step - loss: 0.4304 - accuracy: 0.9800 - recall: 0.9700 - precision: 0.9798

注：数据增强之后产生的数据具有随机性, 每次训练且每个批次训练数据均不相同, 因此模型的复现并不一定能够符合上面的结果

'''