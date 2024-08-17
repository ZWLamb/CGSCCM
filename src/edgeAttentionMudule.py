import torch
import torch.nn as nn


class CFFFModule(nn.Module):
    def __init__(self, encoder_channels):
        super(CFFFModule, self).__init__()

        self.attention_self = nn.ModuleList()
        self.attention_other = nn.ModuleList()
        self.conv_fusion = nn.ModuleList()
        # 注意力机制的参数和层
        for i in range(len(encoder_channels)):
            self.attention_self.append(nn.Conv2d(encoder_channels[i], 1, kernel_size=1).cuda())  #
            self.attention_other.append(nn.Conv2d(encoder_channels[i], 1, kernel_size=1).cuda()) #
            self.conv_fusion.append(nn.Conv2d(encoder_channels[i]*2, encoder_channels[i], kernel_size=1).cuda())  # 用于融合的卷积层



        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)  # 添加ReLU激活函数


    def forward(self, self_features, other_features):
        fused_features_array = []
        for i in range(len(self_features)):
            # 计算自注意力权重
            self_attention_logits = self.attention_self[i](self.relu(self_features[i].clone()))  # 使用卷积层和ReLU激活函数
            self_attention_weights = self.softmax(self_attention_logits)

            # 计算其他编码器注意力权重
            other_attention_logits = self.attention_other[i](self.relu(other_features[i].clone()))  # 使用卷积层和ReLU激活函数
            other_attention_weights = self.softmax(other_attention_logits)

            # 使用注意力权重对特征进行加权求和
            attended_self_features = self_features[i] * self_attention_weights  # 输出通道数与输入通道数相同
            attended_other_features = other_features[i] * other_attention_weights  # 输出通道数与输入通道数相同

            # 融合注意力加权特征
            fused_features = torch.cat((attended_self_features, attended_other_features), dim=1)  # 在通道维度上拼接

            # 使用卷积层调整通道数，使其与原始特征的通道数相同
            fused_features_array.append(self.conv_fusion[i](fused_features))

        return fused_features_array


if __name__ == '__main__':
    # 示例用法
    encoder1_features = torch.randn(2, 64, 10, 10)  # 示例编码器1特征，假设形状为(batch_size, channels, height, width)
    encoder2_features = torch.randn(2, 64, 10, 10)  # 示例编码器2特征，假设形状为(batch_size, channels, height, width)

    # 实例化双向注意力模块
    attention_module1_to_2 = CFFFModule(encoder_channels=64)
    attention_module2_to_1 = CFFFModule(encoder_channels=64)

    # 应用双向注意力模块
    fused_encoder1_features = attention_module1_to_2(encoder1_features, encoder2_features)
    fused_encoder2_features = attention_module2_to_1(encoder2_features, encoder1_features)

    # 输出形状
    print("编码器1融合后的特征形状:", fused_encoder1_features.shape)
    print("编码器2融合后的特征形状:", fused_encoder2_features.shape)
