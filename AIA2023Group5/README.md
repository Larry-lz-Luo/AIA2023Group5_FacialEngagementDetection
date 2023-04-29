# Setup Develop Envirment

Download openvino toolkit and runtime 2022.3.0:
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3/windows/w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64.zip

wget -Uri https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3/windows/w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64.zip -OutFile -openvino_toolkit_windows_2022.3.0.zip

mkdir openvino_toolkit_windows_2022.3.0 
tar -xf openvino_toolkit_windows_2022.3.0.zip -C openvino_toolkit_windows_2022.3.0 --strip-components 1
