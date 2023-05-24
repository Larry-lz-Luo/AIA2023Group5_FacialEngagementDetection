# AIA2023Group5_Facial_Engagement_Detection

專案資料夾架構如下:  

```
├───ModelTraining --> 模型訓練相關
├───POC           --> POC demo 專案 (Win32 C++)
└───PythonSamples --> 其他相關python的測試
```

Taiwan AIA Group 5

成員:

* XT121006 [王傑生](https://github.com/trpleokslab)  
* XT121007 [洪啟豪](https://github.com/Charlie-TW)  
* XT121008 [駱豊儒](https://github.com/Larry-lz-Luo)  
* XT121020 [劉品賢](https://github.com/PXLife)  

本專題為課程公開組題目 : 面部專心識別應用於線上教學的系統 POC 開發

POC demo 基於 OpenVino 的 [Gaze Estimation Demo](https://docs.openvino.ai/latest/omz_demos_gaze_estimation_demo_cpp.html#doxid-omz-demos-gaze-estimation-demo-cpp) 去修改實作完成。

由 Gaze Estimation 中解析出來的面部參數作為資料集的收集來源，
由資料集進行訓練後生成出 XGBooster 的 模型供我們預測專心程度。  
* 模型訓練相關請參考 **ModelTraining** 資料夾的內容  
* Demo 實作請參考 **POC** 資料夾的內容  
* 課程其他相關實作請參考 **PythonSamples**  資料夾的內容  