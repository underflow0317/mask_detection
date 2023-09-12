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

原先模型: AveragePooling2D - Flatten - Dense_128 - Dropout_0.5 - Dense_2_softmax  

為使模型更符合"深度"學習，將原先模型(fat+short)增加Dense layer數目並減少Dense layer中的neuron(thin+tall):  
1. AveragePooling2D - Flatten - Dense_64 - Dense_64 - Dropout_0.5 - Dense_2_softmax  
2. AveragePooling2D - Flatten - Dense_32 - Dropout_0.25 - Dense_32 - Dropout_0.25 - Dense_32 - Dropout_0.25 - Dense_32 - Dropout_0.5 - Dense_2_softmax  

![module_compare](https://github.com/underflow0317/mask_detection/blob/main/module_opt.png)  

可以看出，thin+tall在train size<=2000後，會比fat+short更學習成效。  
所以在模型變得thin+tall時，相較於原先fat+short的模型，當訓練資料較小時有著更高的測試準確率(e.g.,當train_size=2000 or 1000時)，i.e,thin+tall的模型對於未知的(private)資料集能有較佳的準確度。
 
其他應用:  
Cyber Security:  
-Facial Authentication System for the Web:將Upload image改為由鏡頭提供，以至於即時識別身分，可用於線上監考。
