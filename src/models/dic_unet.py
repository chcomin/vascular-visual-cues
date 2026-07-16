import torch
import torch.nn as nn


class DicUNet(nn.Module):
    """U-Net model where kernel sizes, dilation ratios and pooling operations are created from a
    dictionary of arch_conf containing the configuration of each block of the model."""

    def __init__(self, arch_conf):
        super(DicUNet, self).__init__()

        self.n_classes = 1
        # FIRST
        self.first_residual = nn.Sequential(
            nn.Conv2d(**arch_conf["FR"]),
            nn.BatchNorm2d(arch_conf["FR"]["out_channels"])
        )

        self.first_conv = nn.Sequential(
            nn.Conv2d(**arch_conf["FC"][0]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["FC"][0]["out_channels"]),
            nn.Conv2d(**arch_conf["FC"][1]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["FC"][1]["out_channels"])
        )

        # DOWN PATH
        if arch_conf["D1"]["kernel_size"] == 2 and arch_conf["UP2"]["kernel_size"] == 2:
            self.down_pooling1 = nn.MaxPool2d(**arch_conf["D1"])
        else: 
            self.down_pooling1 = nn.Identity()

        self.block1_residual = nn.Sequential(
            nn.Conv2d(**arch_conf["B1R"]),
            nn.BatchNorm2d(arch_conf["B1R"]["out_channels"])
        )
        self.block1_conv = nn.Sequential(
            nn.Conv2d(**arch_conf["B1C"][0]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["B1C"][0]["out_channels"]),
            nn.Conv2d(**arch_conf["B1C"][1]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["B1C"][1]["out_channels"])
        )

        if arch_conf["D2"]["kernel_size"] == 2 and arch_conf["UP1"]["kernel_size"] == 2:
            self.down_pooling2 = nn.MaxPool2d(**arch_conf["D2"])
        else:
            self.down_pooling2 = nn.Identity()
        self.block2_residual = nn.Sequential(
            nn.Conv2d(**arch_conf["B2R"]),
            nn.BatchNorm2d(arch_conf["B2R"]["out_channels"])
        )
        self.block2_conv = nn.Sequential(
            nn.Conv2d(**arch_conf["B2C"][0]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["B2C"][0]["out_channels"]),
            nn.Conv2d(**arch_conf["B2C"][1]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["B2C"][1]["out_channels"])
        )

        # UP PATH
        if arch_conf["D2"]["kernel_size"] == 2 and arch_conf["UP1"]["kernel_size"] == 2:
            self.up_layer1 = nn.ConvTranspose2d(**arch_conf["UP1"])
        else: 
            arch_conf_aux = {
                                "in_channels": arch_conf["B2C"][1]["out_channels"],
                                "out_channels": arch_conf["B1C"][1]["out_channels"],
                                "kernel_size": 1,
                                "padding": 0,
                                "dilation": 1
                            }
            self.up_layer1 = nn.Conv2d(**arch_conf_aux)

        self.conv_bridge1 = nn.Sequential(
            nn.Conv2d(**arch_conf["CB1"]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["CB1"]["out_channels"])
        )
        self.up_block1_residual = nn.Sequential(
            nn.Conv2d(**arch_conf["UB1R"]),
            nn.BatchNorm2d(arch_conf["UB1R"]["out_channels"])
        )
        self.up_block1_conv = nn.Sequential(
            nn.Conv2d(**arch_conf["UB1C"][0]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["UB1C"][0]["out_channels"]),
            nn.Conv2d(**arch_conf["UB1C"][1]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["UB1C"][1]["out_channels"])
        )

        
        # UP PATH
        if arch_conf["D1"]["kernel_size"] == 2 or arch_conf["UP2"]["kernel_size"] == 2:
            self.up_layer2 = nn.ConvTranspose2d(**arch_conf["UP2"])          
        else: 
            arch_conf_aux = {
                                "in_channels": arch_conf["UB1C"][1]["out_channels"],
                                "out_channels": arch_conf["FC"][1]["out_channels"],
                                "kernel_size": 1,
                                "padding": 0,
                                "dilation": 1
                            }
            self.up_layer2 = nn.Conv2d(**arch_conf_aux)

        self.conv_bridge2 = nn.Sequential(
            nn.Conv2d(**arch_conf["CB2"]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["CB2"]["out_channels"])
        )
        self.up_block2_residual = nn.Sequential(
            nn.Conv2d(**arch_conf["UB2R"]),
            nn.BatchNorm2d(arch_conf["UB2R"]["out_channels"])
        )
        self.up_block2_conv = nn.Sequential(
            nn.Conv2d(**arch_conf["UB2C"][0]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["UB2C"][0]["out_channels"]),
            nn.Conv2d(**arch_conf["UB2C"][1]),
            nn.ReLU(),
            nn.BatchNorm2d(arch_conf["UB2C"][1]["out_channels"])
        )

        self.conv_final = nn.Conv2d(**arch_conf["F"])

    def forward(self, x):
        # FIRST
        first_residual = self.first_residual(x)
        first_conv = self.first_conv(x)
        out_first = first_conv + first_residual

        # BLOCK 1
        outpooling1 = self.down_pooling1(out_first)
        block1_residual = self.block1_residual(outpooling1)
        block1_conv = self.block1_conv(outpooling1)
        out_block1 = block1_conv + block1_residual

        # BLOCK 2
        outpooling2 = self.down_pooling2(out_block1)
        block2_residual = self.block2_residual(outpooling2)
        block2_conv = self.block2_conv(outpooling2)
        out_block2 = block2_conv + block2_residual

        # UP BLOCK 1
        out_uplayer1 = self.up_layer1(out_block2)
        out_convBridge1 = self.conv_bridge1(out_block1)
        out_Concat1 = torch.cat([out_uplayer1, out_convBridge1], dim=1)
        up_block1_residual = self.up_block1_residual(out_Concat1)
        up_block1_conv = self.up_block1_conv(out_Concat1)
        out_up_block1 = up_block1_conv + up_block1_residual

        # UP BLOCK 2
        out_uplayer2 = self.up_layer2(out_up_block1)
        out_convBridge2 = self.conv_bridge2(out_first)
        out_Concat2 = torch.cat([out_uplayer2, out_convBridge2], dim=1)
        up_block2_residual = self.up_block2_residual(out_Concat2)
        up_block2_conv = self.up_block2_conv(out_Concat2)
        out_up_block2 = up_block2_conv + up_block2_residual

        final = self.conv_final(out_up_block2)
        return final
