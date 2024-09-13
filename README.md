# arcface
A face recognition model

## 执行

1. 在开发板上安装 Python 执行环境。注意：
   - RKNN-Toolkit Ubuntu 20.04 只支持 Python 3.8 和 3.9。
   - Ubuntu 22.04 支持 Python 3.10 和 3.11。

2. 在开发板上执行以下命令以安装必要的库：
   ```bash
   sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc
   pip install -i https://mirror.baidu.com/pypi/simple opencv_contrib_python
   ```

3. 获得瑞芯微官方的 RKNN-Toolkit：
   ```bash
   git clone https://hub.nuaa.cf/airockchip/rknn-toolkit2.git
   ```

4. 将 RKNN-Toolkit 安装到开发板上：
   ```bash
   cd rknn-toolkit2
   pip install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.0.0b0-cpxx-cpxx-linux_aarch64.whl
   ```
   其中的 `cpxx` 需要替换为 Python 的版本号，请在 `rknn-toolkit-lite2/packages` 目录下查找和你的 Python 版本匹配的 `.whl` 文件。例如：
   ```bash
   pip install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.0.0b0-cp38-cp38-linux_aarch64.whl
   ```

5. 将本项目拷贝至 RK3588 开发板。

6. 执行：
   ```bash
   python inference_onnx.py
   ```

## 模型转换

1. 本项目原始模型来自 [arcface-pytorch](https://github.com/bubbliiiing/arcface-pytorch)。

2. 如果需要重新生成模型，请执行以下步骤：

3. clone https://github.com/bubbliiiing/arcface-pytorch.git
   
4. 拷贝 pth2pt.py 至 arcface-pytorch 根目录，并执行：
   ```bash
   python pth2pt.py
   ```
   生成 TorchScript 模型文件 `arcface.pt`。

5. 将生成的 `arcface.pt` 拷贝至本项目 `model` 目录下，执行：
   ```bash
   python pt2onnx.py
   ```
   生成 ONNX 文件 `arcface.onnx`。

6. 如果需要继续生成 rknn 模型，请参考以下步骤。但目前生成的 arface rknn 模型在 rk3588 上无法得到有意义的结果，不确定是否因为精度不足。
   
7. 参考 [RKNN-Toolkit2 快速入门指南](https://hub.nuaa.cf/airockchip/rknn-toolkit2/blob/master/doc/01_Rockchip_RV1106_RV1103_Quick_Start_RKNN_SDK_V2.0.0beta0_CN.pdf)，在 PC 端用 Docker 安装 RKNN-Toolkit2 镜像。

8. 启动镜像后，在镜像中执行以下步骤：

9.  利用 `docker cp` 将本项目源码拷贝进容器中。

10. 准备量化样本。本例中使用从 [faces_ms1m_112x112.zip](https://s3.amazonaws.com/onnx-model-zoo/arcface/dataset/faces_ms1m_112x112.zip) 下载的人脸图片集，并解压缩至 dataset 目录。
    
11. 执行：
    ```bash
    python ms_celeb_1m_tool.py
    ```
    提取图片文件。目前只取了 lfw 图片。如果需要提取其他文件，请修改 ms_celeb_1m_tool.py

12. 编辑 `gen_ds.py`，将路径指向样本集根目录。执行：
    ```bash
    python gen_ds.py
    ```
    获得 `dataset.txt` 文件。

13. 执行：
    ```bash
    python3 pt2rknn.py <model_name>.pt <platform>
    ```
    例如：
    ```bash
    python3 pt2rknn.py model/arcface.pt rk3588
    ```
    之后即可获得 `arcface.rknn` 模型文件。

## ONNX

如果希望通过 ONNX 格式进行模型转换，执行 `python onnx2rknn.py`。参数请在文件内修改。

## 关于参数

输入需归一化到 [−1, 1]：mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]]

## 参考

- 另一个 Arcface 模型可见： [Arcface](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface)。该模型比较大，但其量化版本也许可以尝试转化为 rknn。其输入需归一化至 [0, 2]