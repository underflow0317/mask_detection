face_detection(res10_300x300_ssd_iter_140000.caffemodel) is from: https://github.com/keyurr2/face-detection/tree/master

steps:
1. 用 opt_train.py 產生 mask detection module
2. 執行
   2-1. web_image/app.py  
       2-1-1. Select and Uploads image  
       2-1-2. Flask 呼叫 detect_mask_image.py to detect  
       2-1-3. Result 將回傳至Flask並呈現在web
     
   2-2. web_video/app.py  
       2-2-1. flask透過鏡頭即時顯示result，利用threading更新fps  
   

使用mobilenet_v2  
在test_size=0.1下，  
原先模型(AveragePooling2D - Flatten - Dense_128 - Dropout_0.5 - Dense_2_softmax)得:  
    test_size=0.1:  
       -acc=0.9913  
       -test_acc=0.9913  
   test_size=0.3:  
       -acc=0.9854  
       -test_acc=0.9765  
  
為使模型更符合"深度"學習，將原先模型(fat+short)增加Dense layer數目並減少Dense layer中的neuron(thin+tall):  
   AveragePooling2D - Flatten - Dense_64 - Dropout_0.25 - Dense_32 - Dropout_0.5 - Dense_2_softmax  
並得:  
   test_size=0.1:  
       -acc=0.9826  
       -test_acc=0.9739   
   test_size=0.3:  
       -acc=0.9795  
       -test_acc=0.9843  
  
在模型變得thin+tall時，相較於原先fat+short的模型，當訓練資料較小時有著更高的測試準確率(e.g.,當test_size=0.3時)，i.e,thin+tall的模型對於未知的(private)資料集能有較佳的準確度。
 
其他應用:  
Cyber Security:  
-Facial Authentication System for the Web:將Upload image改為由鏡頭提供，以至於即時識別身分，可用於線上監考
