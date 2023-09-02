from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np
import os
import random
import tensorflow as tf
import pickle # serialization
#from compress_pickle import dump, load

physical_device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
bce = tf.keras.losses.BinaryCrossentropy()

def Resize_And_Convert_To_Numpy_Array(pic,width=224,height=224):

  image = load_img(pic, target_size=(width, height))
  image = img_to_array(image)
  image = preprocess_input(image)

  return  image
pass


class CustomModule(tf.Module):

  def __init__(self, i_x_size=224, i_y_size=224, i_z_size=3, i_num_of_types=2):
    super(CustomModule, self).__init__()
    self.opt =tf.keras.optimizers.Adam(learning_rate=0.0001)
    self.x_size =i_x_size
    self.y_size =i_y_size
    self.z_size =i_z_size #rgb
    self.num_of_types=i_num_of_types # 0: no mask, 1: mask
    self.tf_map_op={}
    self.tf_map_var={}
    self.init( )

  pass

  # can only init variables by python function, not tf function
  def init(self):

    
    self.tf_map_var['W1'] = tf.Variable(tf.random.truncated_normal([5, 5, self.z_size, 32], stddev=0.01))  
    self.tf_map_var['b1'] = tf.Variable(tf.zeros(32))  

    self.tf_map_var['W2'] = tf.Variable(tf.random.truncated_normal([6, 6, 32, 64], stddev=0.01))  
    self.tf_map_var['b2'] = tf.Variable(tf.zeros(64))  

    self.tf_map_var['W3'] = tf.Variable(tf.random.truncated_normal([5, 5, 64, 64], stddev=0.01))  
    self.tf_map_var['b3'] = tf.Variable(tf.zeros(64)) 

    self.tf_map_var['W4'] = tf.Variable(tf.random.truncated_normal([10816, 1024], stddev=0.01))  
    self.tf_map_var['b4'] = tf.Variable(tf.zeros(1024))  

    self.tf_map_var['W5'] = tf.Variable(tf.random.truncated_normal([1024, self.num_of_types], stddev=0.01))  
    self.tf_map_var['b5'] = tf.Variable(tf.zeros(self.num_of_types))    
  pass
  
  
  def train(self, y , x):
    L=0
    with tf.GradientTape() as tape:
      prediction=self.predict(x)
      L = self.get_loss( y, prediction)
      g = tape.gradient(L, self.tf_map_var.values())
    pass
    result=zip(g,  self.tf_map_var.values())
    self.opt.apply_gradients(result)  
    return L
  pass
  

  def get_loss( self, y, prediction):
    return tf.reduce_mean(-tf.reduce_sum(tf.constant(y, dtype=tf.float32) * tf.math.log(prediction),axis=[1])) # loss
  pass
  

  @tf.function(input_signature=[tf.TensorSpec([None,224,224,3], tf.float32)])
  def predict(self, input_x):
 
    self.tf_map_op['conv1'] = tf.nn.relu(tf.add(tf.nn.conv2d(input_x, self.tf_map_var['W1'], strides=[1, 2, 2, 1], padding='SAME') , self.tf_map_var['b1']))
    #print("conv1:", conv1.shape)
    self.tf_map_op['conv2'] = tf.nn.max_pool(self.tf_map_op['conv1'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    self.tf_map_op['conv3'] = tf.nn.relu(tf.add(tf.nn.conv2d(self.tf_map_op['conv2'], self.tf_map_var['W2'], strides=[1, 2, 2, 1], padding='VALID') , self.tf_map_var['b2']))
    #print("conv3:", conv3.shape)
    self.tf_map_op['conv4'] = tf.nn.relu(tf.add(tf.nn.conv2d(self.tf_map_op['conv3'], self.tf_map_var['W3'], strides=[1, 2, 2, 1], padding='SAME') , self.tf_map_var['b3']))
    #print("conv4:", conv4.shape)

    self.tf_map_op['conv5'] = tf.reshape(self.tf_map_op['conv4'], [-1, 10816]) 
    #print("conv5:", conv5.shape)
    self.tf_map_op['conv6'] = tf.nn.relu(tf.add(tf.matmul(self.tf_map_op['conv5'], self.tf_map_var['W4']) , self.tf_map_var['b4']))
    #print("conv6:", conv6.shape)
    self.tf_map_op['conv6'] = tf.nn.softmax(tf.add(tf.matmul(self.tf_map_op['conv6'], self.tf_map_var['W5']) , self.tf_map_var['b5']))
    return self.tf_map_op['conv6']
  pass
pass  


def main():

  ############################
  # 0. environment setting
  
  # location of picture
  picdir= "./TrainData/"
  
  # size of picture
  width_size=224
  height_size=224

  # init variables to store sample
  x_data = []
  y_labels = []
  
  # init tensorflow variables 
  model = CustomModule(width_size, height_size)
  

  
  ############################
  # 1. loading picture
  
  for dirPath, dirNames, fileNames in os.walk(picdir):
      for f in fileNames:               
          # get y (ans)
          ans=int(dirPath[len(picdir):])
          
          # get x (picture)
          PicDir= os.path.join(dirPath, f)
          NArray_2D  = Resize_And_Convert_To_Numpy_Array(PicDir,width_size,height_size)  
          
          # store y, x 
          y_labels.append(ans)
          x_data.append(NArray_2D)
      pass
  pass
  
  # save sample to file


  f = open('pic_bw_db_224_224.p', 'wb')
  pickle.dump([y_labels,x_data], f)
  f.close()
  
  
  # load sample from file
  
  f = open('pic_bw_db_224_224.p', 'rb')
  r=pickle.load(f)
  y_labels=r[0]
  x_data=r[1]
  
  # convert the data and labels to NumPy arrays
  x_data = np.array(x_data, dtype="float32")
  y_labels = np.array(y_labels)


  # 4.1.2. 利用tensorflow.keras.utils.to_categorical
  #        將binary編碼轉成one hot encoding
  #        one hot encoding可以作多重分類的訓練之用
  #        例如：
  #
  # 有如下三個特徵屬性：
  # · 性別：["male"，"female"]
  # · 地區：["Europe"，"US"，"Asia"]
  # · 流覽器：["Firefox"，"Chrome"，"Safari"，"Internet Explorer"]
  #  對於某一個樣本，如["male"，"US"，"Internet Explorer"]，
  #  我們需要將這個分類值的特徵數位化，最直接的方法，我們可以採用序列化
  #  的方式：[0,1,3]。但是這樣的特徵處理並不能直接放入機器學習演算法中。

  # 採用One-Hot編碼的方式對上述的樣本
  # “["male"，"US"，"Internet Explorer"]”編碼，
  # “male”則對應著[1，0]，同理“US”對應著[0，1，0]，
  # “Internet Explorer”對應著[0,0,0,1]。
  # 則完整的特徵數位化的結果為：[1,0,0,1,0,0,0,0,1]。
  lb = LabelBinarizer()
  y_labels2 = lb.fit_transform(y_labels)

  y_labels3 = to_categorical(y_labels2,2) 
  
  
  # 4.2. 使用sklearn.model_selection將資料分割成訓練集與測試集
  #      X => 圖檔
  #      Y => 答案
  # train_data：所要劃分的樣本特徵集 X 
  # train_target：所要劃分的樣本結果 Y
  # test_size：多少樣本拿去測試 0.25=> 25% 拿去測試
  # stratify: 25%是根據那一個比例，在這裡是針對labels的比例，亦即mask的25%以及unmask的25%
  # random_state：隨機數的種子。
  (trainX, testX, trainY, testY) = train_test_split(x_data, y_labels3,
      test_size=0.1, stratify=y_labels3, random_state=42)

  # construct the training image generator for data augmentation
  aug = ImageDataGenerator(
    rotation_range=20, # 角度值，0~180，影象旋轉
    zoom_range=0.15, # 隨機縮放範圍
    width_shift_range=0.2,  # 垂直平移，相對總高度的比例
    height_shift_range=0.2, # 水平平移，相對總寬度的比例
    shear_range=0.15, # 隨機錯切換角度
    horizontal_flip=True, # 一半影象水平翻轉
    fill_mode="nearest")
    
  ############################
  # 2. training
  NumberOfOneTraining = 32 # batch size
  Training_times= 100 #訓練次數

  for i in range(Training_times): 
     
    ##############################
    # get training data randomly
    
    # init x batch and y batch
    x_batch=[]
    y_batch=[]
    
    sample = random.sample(range(len(trainY)), NumberOfOneTraining)
  
    x_batch = [trainX[i] for i in sample]
    x_batch = tf.constant(x_batch, shape=[NumberOfOneTraining,model.x_size,model.y_size,model.z_size], dtype=tf.float32) 
    y_batch = [trainY[i] for i in sample]

    ##############################
    # training
    
    model.train(y_batch,x_batch)
    
    
    
    # training the package
    #opt.apply_gradients(package)
    

    ##############################
    # test the test data set per training
    '''
    num_error=0
    total=0
    _loss=0.0
    for j in range(len(testY)):
        input_batch=1
        input_x=tf.constant(testX[j], shape=[input_batch,model.x_size,model.y_size,model.z_size], dtype=tf.float32) 
        prediction=model.predict(input_x)
        _loss=_loss+model.get_loss(testY[j],prediction).numpy()
        if np.argmax(testY[j]) != np.argmax(prediction) :
              num_error=num_error+1
        pass
        total=total+1
    pass
    accuracy_avg=(1.0-num_error/total)
    loss_avg=_loss/total
    print("train, {} , accuracy, {}, loss avg, {} ".format( i,accuracy_avg,loss_avg), flush=True)
    '''
  pass
  print("#### training finished ")
    
  ##############################
  # test the test data set
  
  num_error=0
  total=0
  for j in range(len(testY)):
      input_batch=1
      input_x=tf.constant(testX[j], shape=[input_batch,model.x_size,model.y_size,model.z_size], dtype=tf.float32) 
      prediction=model.predict(input_x)
      if np.argmax(testY[j]) != np.argmax(prediction) :
            num_error=num_error+1
      pass
      total=total+1
  pass
  accuracy_avg=(1.0-num_error/total)
  print("fianl test accuracy, {}".format( accuracy_avg), flush=True)
  
  ##############################
  # save model

  tf.saved_model.save(model, "masked_module")
  
pass

if __name__ == '__main__':
  main()  # entry function
pass
