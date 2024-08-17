import torch
# import segmentation_models_pytorch as smp
from src.base_net import UNet, build_encoder, UnetDecoder, OutLayer
import torch.nn as nn
from src.edgeAttentionMudule import CFFFModule


class SC_Net(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.model_name = "unet"
        self.classes = 4
        self.activaton = None
        self.device = device

        self.model = None
        if self.model_name == "unet":
            self.encoder0 = build_encoder()
            self.encoder1 = build_encoder()

            self.decoder0 =  UnetDecoder(
            encoder_channels=self.encoder0.out_channels,
            decoder_channels= (256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            attention_type="scse",
        )
            self.decoder1 =  UnetDecoder(
            encoder_channels=self.encoder0.out_channels,
            decoder_channels= (256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            attention_type="scse",
        )

            self.decoder2 = UnetDecoder(
                encoder_channels=self.encoder1.out_channels,
                decoder_channels=(256, 128, 64, 32, 16),
                n_blocks=5,
                use_batchnorm=True,
                attention_type="scse",
            )

            self.out_layer0 =  OutLayer(
            in_channels=16,
            out_channels=2,
            activation=self.activaton,
            kernel_size=3,
        )
            self.out_layer1 = OutLayer(
            in_channels=16,
            out_channels=1,
            activation=self.activaton,
            kernel_size=3,
        )
            self.out_layer2 =  OutLayer(
            in_channels=16,
            out_channels=1,
            activation=self.activaton,
            kernel_size=3,
        )

            self.attention_module1_to_2 = CFFFModule(encoder_channels=[3, 24, 48, 64, 160, 256])

        else:
            print("Model Not Found !")

    def forward(self, x):
        features0 = self.encoder0(x)  # 3
        features1 = self.encoder1(x)  # 3


        fused_encoder1_features = self.attention_module1_to_2(features0, features1)  # input:features0, features1 list


        decoder_output = self.decoder0(*fused_encoder1_features)  # 16
        decoder_output1 = self.decoder1(*fused_encoder1_features)  # 16
        decoder_output2 = self.decoder2(*features1)  # 16

        decoder_output = self.out_layer0(decoder_output)  # 2 bg
        decoder_output1 = self.out_layer1(decoder_output1)  # 1 cp
        decoder_output2 = self.out_layer2(decoder_output2)  # 1 cf

        masks = torch.zeros(
            decoder_output1.size(0),
            self.classes,
            decoder_output1.size(2),
            decoder_output1.size(3),
        ).to(self.device)
        masks[:, :2, :, :] = decoder_output  # 2
        masks[:, 2, :, :] = decoder_output1.squeeze()
        masks[:, 3, :, :] = decoder_output2.squeeze()

        return masks

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        x = self.forward(x)
        return x



if __name__ == '__main__':
    from torchsummary import summary

    from thop import profile

    model = SC_Net("cuda").cuda()
    #summary(model, input_size=(1, 3, 256, 256))
    input = torch.randn(1,3,256,256).cuda()
    flops,params = profile(model,(input,))
    print(flops)
    print("flops:%.2f B,params:%.2f M" % (flops / 1e9,params / 1e6))

    import time
    torch.cuda.synchronize()
    start = time.time()
    result = model(input)
    torch.cuda.synchronize()
    end = time.time()
    print(end-start)