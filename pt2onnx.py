import torch

# 加载模型
model = torch.jit.load('./model/arcface.pt')
model.eval()  # 设置模型为评估模式

# 假设模型输入是一个形状为 (batch_size, num_channels, height, width) 的张量
dummy_input = torch.randn(1, 3, 112, 112)

torch.onnx.export(
    model,                     # 被保存的模型
    dummy_input,               # 示例输入，用于构建计算图
    "./model/arcface.onnx",              # 输出文件名
    export_params=True,        # 存储模型权重
    opset_version=19,          # 你可以选择一个合适的 ONNX opset 版本（更高版本可能支持更多的特性）
    do_constant_folding=True,  # 真正执行常量折叠优化以提高性能
    input_names=['input'],     # 模型输入的名字
    output_names=['output'],   # 模型输出的名字
    dynamic_axes={
        'input': {0: 'batch_size'},    # 动态声明输入的批量维度
        'output': {0: 'batch_size'}    # 动态声明输出的批量维度
    }
)