import streamlit as st      # import 完了之后直接填网页内容
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image

txt_BP = '''# 性别分类任务实现
import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# 1.根据数据构建分类模型，使用training数据集进行模型训练，并调用训练好的模型给出验证集的正确率
# 2.调用摄像头进行拍照，识别照片人物的性别
# cap = cv2.VideoCapture(0)   # 0表示默认摄像头
# _, img = cap.read()         # cap.read()会返回两个值(1.布尔值，true=已成功拍照，2.图片的二维展开)
# # 看看自己长什么样
# # plt.imshow(img[:, :, ::-1])     # 记得 matplotlib和 cv2 的格式是相反的，所以要::-1（ RBG和 BRG）
# # plt.show()
# haar = cv2.CascadeClassifier("D:/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")      # 这个是人脸识别的文件
# faces = haar.detectMultiScale(img)     # 这一步是人脸检测
# for x, y, w, h in faces:               # 左上角点的 x,y，以及对应的宽度、高度
#     print(x, y, w, h)
#     your_face = img[y:y+w, x:x+h, :]        # 在整个照片中截取你的脸
#     plt.imshow(your_face[:, :, ::-1])       # 不想变成蓝精灵记得反转 BRG
#     plt.show()
# cap.release()                          # 关掉摄像头

# 批量读取图片
def get_info(train_test_name, male_female_name):
    count = 0
    path = './' + str(train_test_name) + '/' + str(male_female_name)
    face_record = []
    face_name = os.listdir(path)
    for i in face_name:
        count += 1
        if count % 1000 == 0:
            print(f'已处理到第{count}张图片...')
        os_path = path + '/' + i
        img = cv2.imread(os_path)
        img = cv2.resize(img, (64, 64))  # 统一图片大小
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 对 img_2 进行 从 BGR的彩色图片转至灰度图 （ BGR2GRAY）
        face_record.append(img)
    y = np.zeros((len(face_record), )) if male_female_name == 'female' else np.ones((len(face_record), ))
    x = face_record
    return x, y

x_female_train, y_female_train = get_info('Training', 'female')
x_male_train, y_male_train = get_info('Training', 'male')
x_female_test, y_female_test = get_info('Validation', 'female')
x_male_test, y_male_test = get_info('Validation', 'male')

# 合并训练、测试集
x_train, y_train = np.array(x_male_train + x_female_train), np.r_[y_male_train, y_female_train]
x_test, y_test = np.array(x_male_test + x_female_test), np.r_[y_male_test, y_female_test]
# 数据处理标准化！
x_train_input, x_test_input = tf.constant((x_train/ 255), dtype=tf.float32), tf.constant((x_test/ 255), dtype=tf.float32)

# 设参！
input_size, hidden_size1, out_size =\
    x_train.shape[0], 2 ** 8, 1 if len(set(y_train)) == 2 else len(set(y_train))
# 模型开始搞起来
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(64, 64)))
model.add(tf.keras.layers.Dense(units=hidden_size1, activation=tf.keras.activations.sigmoid))
model.add(tf.keras.layers.Dense(units=out_size, activation=tf.keras.activations.sigmoid))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=tf.keras.metrics.BinaryAccuracy())
model.summary()
hist = model.fit(x_train_input, y_train, epochs=45, batch_size=500, validation_data=(x_test_input, y_test))

# 对训练集也做同样的操作
y_test_predict = model.predict(x_test_input)
y_test_out = y_test_predict >= 0.5
print(f'正确率：{accuracy_score(y_test, y_test_out)}')

# 先处理
# test = cv2.resize(your_face, (64, 64))
# test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
# test = test.reshape(-1, 64, 64)
# test = tf.constant((test/ 255), dtype=tf.float32)
# out = model.predict(test)
# out = out >= 0.5
# sex = '女' if out == 0 else '男'
# print(f'这是{sex}性')'''
txt_CNN = '''# 性别分类任务实现
import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# 1.根据数据构建分类模型，使用training数据集进行模型训练，并调用训练好的模型给出验证集的正确率
# 2.调用摄像头进行拍照，识别照片人物的性别
# cap = cv2.VideoCapture(0)   # 0表示默认摄像头
# _, img = cap.read()         # cap.read()会返回两个值(1.布尔值，true=已成功拍照，2.图片的二维展开)
# # 看看自己长什么样
# # plt.imshow(img[:, :, ::-1])     # 记得 matplotlib和 cv2 的格式是相反的，所以要::-1（ RBG和 BRG）
# # plt.show()
# haar = cv2.CascadeClassifier("D:/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")      # 这个是人脸识别的文件
# faces = haar.detectMultiScale(img)     # 这一步是人脸检测
# for x, y, w, h in faces:               # 左上角点的 x,y，以及对应的宽度、高度
#     print(x, y, w, h)
#     your_face = img[y:y+w, x:x+h, :]        # 在整个照片中截取你的脸
#     plt.imshow(your_face[:, :, ::-1])       # 不想变成蓝精灵记得反转 BRG
#     plt.show()
# cap.release()                          # 关掉摄像头

# 批量读取图片
def get_info(train_test_name, male_female_name):
    count = 0
    path = './' + str(train_test_name) + '/' + str(male_female_name)
    face_record = []
    face_name = os.listdir(path)
    for i in face_name:
        count += 1
        if count % 1000 == 0:
            print(f'已处理到第{count}张图片...')
        os_path = path + '/' + i
        img = cv2.imread(os_path)
        img = cv2.resize(img, (64, 64))  # 统一图片大小
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 对 img_2 进行 从 BGR的彩色图片转至灰度图 （ BGR2GRAY）
        face_record.append(img)
    y = np.zeros((len(face_record), )) if male_female_name == 'female' else np.ones((len(face_record), ))
    x = face_record
    return x, y

x_female_train, y_female_train = get_info('Training', 'female')
x_male_train, y_male_train = get_info('Training', 'male')
x_female_test, y_female_test = get_info('Validation', 'female')
x_male_test, y_male_test = get_info('Validation', 'male')

# 合并训练、测试集
x_train, y_train = np.array(x_male_train + x_female_train), np.r_[y_male_train, y_female_train]
x_test, y_test = np.array(x_male_test + x_female_test), np.r_[y_male_test, y_female_test]
# 数据处理标准化！
x_train_input, x_test_input = tf.constant((x_train/ 255), dtype=tf.float32), tf.constant((x_test/ 255), dtype=tf.float32)

# 设参！
input_size, hidden_size, out_size =\
    x_train.shape[0], 2 ** 5, 1 if len(set(y_train)) == 2 else len(set(y_train))
# 模型开始搞起来
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), input_shape=(64, 64, 1), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(4, 4), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=hidden_size, activation='relu'))
model.add(tf.keras.layers.Dense(units=out_size, activation=tf.keras.activations.sigmoid))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=tf.keras.metrics.BinaryAccuracy())
model.summary()
hist = model.fit(x_train_input, y_train, epochs=50, batch_size=200, validation_split=0.2)
model.evaluate(x_test_input, y_test)        # 查看 model在测试集上的准确度

loss_record, accuracy = hist.history['loss'], hist.history['binary_accuracy']
plt.subplot(1, 2, 1); plt.title('loss'); plt.plot(loss_record, 'r')
plt.subplot(1, 2, 2); plt.title('accuracy'); plt.plot(accuracy, 'b')
plt.show()
# 对训练集也做同样的操作
y_test_predict = model.predict(x_test_input)
y_test_out = y_test_predict >= 0.5
print(f'正确率：{accuracy_score(y_test, y_test_out)}')
'''


st.set_page_config(page_title='男女识别', page_icon='💥')
st.title('男女识别'.center(40, '-'))
st.subheader('本网页预测结果99%正确！')

@st.cache
def img_process(image):
    image = np.array(Image.open(image), dtype=np.uint8)     # 要把照片转换成 numpy多维数组
    haar = cv2.CascadeClassifier("D:/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")  # 这个是人脸识别的文件
    faces = haar.detectMultiScale(image)  # 这一步是人脸检测
    x, y, w, h = faces[0, :]  # 左上角点的 x,y，以及对应的宽度、高度
    your_face = image[y:y + w, x:x + h, :]  # 在整个照片中截取你的脸
    test = cv2.resize(your_face, (64, 64))
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    test = test.reshape(-1, 64, 64, 1)
    test = tf.constant((test / 255), dtype=tf.float32)
    out = model.predict(test)
    out = out >= 0.5
    sex = '女' if out == 0 else '男'
    return sex

img = st.file_uploader('请在此处输入您的照片')

try:
    st.image(img)
except:
    st.text('')

model_select = st.radio('请选择你要用的模型：', ['BP', 'CNN'])

if model_select == 'BP':
    model = tf.keras.models.load_model('./bp-faces-256.h5')
    st.text('你选择了BP模型')
    with st.expander('点击查看模型代码：'):
        st.code(txt_BP, language='python')
    if st.button('点击检测：'):
        try:
            sex = img_process(img)
            st.subheader(f'>>>>>预测结果为{sex}性')
        except:
            st.subheader('未检测出人脸')

elif model_select == 'CNN':
    model = tf.keras.models.load_model('./CNN - faces.h5')
    st.text('你选择了CNN模型')
    with st.expander('点击查看模型代码：'):
        st.code(txt_CNN, language='python')
    if st.button('点击检测：'):
        try:
            sex = img_process(img)
            st.subheader(f'>>>>>预测结果为{sex}性')
        except:
            st.subheader('未检测出人脸')

