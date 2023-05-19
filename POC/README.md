# 設定開發環境
本專案主要使用 Visual Studio 進行編譯，請自行先安裝相關編譯環境。  
https://visualstudio.microsoft.com/zh-hant/downloads/  

以下為第三方環境建置說明:  
Step 1: 
* 下載 openvino toolkit 與 runtime 2022.3.0 並解壓縮至 專案目錄的 includes/openvino_toolkit_windows_2022.3.0 資料夾中
* 下載 openCV4.7.0 並解壓縮至 專案目錄的 includes/opencv 資料夾中
* 這段過程會比較花費時間，因為會下載openvino 與 openCV

可直接執行 step1.bat 或根據需求執行以下指令:
```
wget -Uri https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3/windows/w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64.zip -OutFile openvino_toolkit_windows_2022.3.0.zip

mkdir includes/openvino_toolkit_windows_2022.3.0 
tar -xf openvino_toolkit_windows_2022.3.0.zip -C includes/openvino_toolkit_windows_2022.3.0 --strip-components 1

wget -Uri https://github.com/opencv/opencv/releases/download/4.7.0/opencv-4.7.0-windows.exe -OutFile opencv-4.7.0-windows.exe
opencv-4.7.0-windows.exe -y -o"includes"
```

Step 2:
* 請先安裝好Python 3.10以上
* 安裝openvino 開發工具套件 openvino-dev 
* 使用omz_downloader 根據 models/models.lst 下載需要的預訓練模型
* 使用omz_converter 根據 models/models.lst 對模型進行轉檔

可直接執行 step2.bat 或根據需求執行以下指令:
```
cd models
pip install openvino-dev
pip install --upgrade pip
pip install
omz_downloader --list models.lst
omz_converter --list models.lst
```

以上環境設定完畢專案資料夾結構如下圖:
```
├───AIA2023Group5  
├───faceDB   
├───includes  
│   ├───gazeEstimation  
│   ├───opencv --> 透過step 1 產生  
│   ├───openvino_toolkit_windows_2022.3.0 --> 透過step 1 產生  
│   ├───ovCommon  
│   ├───ovlibs  
│   ├───xgboost  
│   └───xglibs  
├───models  
│   ├───intel --> 透過step 2 產生  
│   └───public --> 透過step 2 產生  
├───pics  
└───Scene2  
```

# AIA2023Group5   

上述開發環境設定完成後，使用 Visual Studio 2022 開啟 AIA2023Group5.sln 應可正常編譯與執行 AIA2023Group5 專案  
進入 Scene2 場景電腦應需要具備有網路能力，因為要讀取 https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_20mb.mp4 影片來模擬會議跟上課看到的畫面。


### POC 架構設計:  
![](pics/POC%20Architecture.png)  

### 註冊人臉資料說明:  
1. 執行POC 程式， Scence1 登入頁面，確認相機啟動有拍攝到自己的正面。
2. 於臉部為清晰正面的狀態下在頁面右上方的 Sign up 按鈕按下進行拍照。
3. 此時應該會有對話框跳出，輸入英文名字或代號按下註冊按鈕進行註冊。
4. 按下取消則可以取消註冊的動作
  
    


# Scene2 錄製資料集說明

使用 Visual Studio 2022 開啟 AIA2023Group5.sln 並執行 Scene2 專案

執行 Scene2.exe 
![](pics/Record1.png)

* Record Concentration :開始錄製專注上課特徵資料，左上方會有錄製提示，再按一次停止。 
![](pics/Record2.png)

* Record Not Concentration :開始錄製非專注上課特徵資料，左上方會有錄製提示，再按一次停止。 
![](pics/Record3.png)
* 錄製檔案csv 儲存於 執行檔案同一目錄下 檔案名稱為開始錄製時間。 
![](pics/Record4.png)