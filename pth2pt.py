# 拷贝本文件至目标模型项目，并修改YourModelClass类为目标模型类

import torch

from arcface import Arcface

# 加载模型权重
model = Arcface()
model.net.eval()  # 切换到评估模式

# # 创建一个输入张量作为示例输入
example_input = torch.randn(1, 3, 112, 112)  # 根据模型输入尺寸调整

# # 将模型转换为 TorchScript
traced_script_module = torch.jit.trace(model.net, example_input, strict=False)

# 保存 TorchScript 模型
torch.jit.save(traced_script_module, "./model/arcface.pt")