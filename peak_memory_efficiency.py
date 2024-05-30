import torch
from ofa.nas.efficiency_predictor import BaseEfficiencyModel
import torch.nn as nn
import copy

from ofa.utils.layers import (
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    MBConvLayer,
    ResidualBlock,
    ZeroLayer,
)

from ofa.classification.elastic_nn.networks import OFAMobileNetV3
from ofa.classification.networks import MobileNetV3


# This class is used to predict the peak memory usage of a given architecture
class PeakMemoryEfficiency(BaseEfficiencyModel):
    def count_peak_activation_size(self, net, data_shape, get_hist=False):
        activation_hist = []

        def record_in_out_size(m, x, y):
            x = x[0]
            m.input_size = torch.Tensor([x.numel()])
            m.output_size = torch.Tensor([y.numel()])

        def add_io_hooks(m_):
            m_type = type(m_)
            if m_type in [nn.Conv2d, nn.Linear]:
                m_.register_buffer("input_size", torch.zeros(1))
                m_.register_buffer("output_size", torch.zeros(1))
                m_.register_forward_hook(record_in_out_size)

        def count_conv_mem(m):
            if m is None:
                return 0
            if hasattr(m, "conv"):
                m = m.conv
            elif hasattr(m, "linear"):
                m = m.linear
            assert isinstance(m, (nn.Conv2d, nn.Linear))
            weight_size = sum(p.numel() for p in m.parameters())
            # print(m)
            # print("Input size is " + str(m.input_size.item()))
            # print("Output size is " + str(m.output_size.item()))
            # print("Weight size is " + str(weight_size))
            # print("Total size is " + str(m.input_size.item() + m.output_size.item() + weight_size))
            return m.input_size.item() + m.output_size.item() + weight_size

        def count_block(m, get_list=False):
            assert isinstance(m, ResidualBlock)

            if m.conv is None or isinstance(m.conv, ZeroLayer):
                return 0

            assert isinstance(m.conv, MBConvLayer)

            if m.shortcut is None or isinstance(m.shortcut, ZeroLayer):
                if get_list:
                    return [
                        count_conv_mem(m.conv.inverted_bottleneck),
                        count_conv_mem(m.conv.depth_conv),
                        count_conv_mem(m.conv.point_linear),
                    ]
                else:
                    return max(
                        [
                            count_conv_mem(m.conv.inverted_bottleneck),
                            count_conv_mem(m.conv.depth_conv),
                            count_conv_mem(m.conv.point_linear),
                        ]
                    )
            else:
                if m.conv.inverted_bottleneck is not None:
                    residual_size = m.conv.inverted_bottleneck.conv.input_size.item()
                else:
                    residual_size = m.conv.depth_conv.conv.input_size.item()

                if get_list:
                    return [
                        count_conv_mem(m.conv.inverted_bottleneck),
                        count_conv_mem(m.conv.depth_conv) + residual_size,
                        count_conv_mem(m.conv.point_linear),
                    ]
                else:
                    return max(
                        [
                            count_conv_mem(m.conv.inverted_bottleneck),
                            count_conv_mem(m.conv.depth_conv) + residual_size,
                            count_conv_mem(m.conv.point_linear),
                        ]
                    )

        if isinstance(net, nn.DataParallel):
            net = net.module
        net = copy.deepcopy(net)

        assert isinstance(net, MobileNetV3)

        net.apply(add_io_hooks)

        with torch.no_grad():
            _ = net(torch.randn(*data_shape).to(next(net.parameters()).device))

        mem_list = [
            count_conv_mem(net.first_conv),
            count_conv_mem(net.final_expand_layer),
            count_conv_mem(net.feature_mix_layer),
            # See below explanation for why we don't keep it
            #count_conv_mem(net.classifier),
        ] + [count_block(blk) for blk in net.blocks]

        activation_hist.append(count_conv_mem(net.first_conv))
        for blk in net.blocks:
            activation_hist += [
                count for count in count_block(blk, get_list=True) if count != 0
            ]
        activation_hist.append(count_conv_mem(net.final_expand_layer))
        activation_hist.append(count_conv_mem(net.feature_mix_layer))
        # I decide to NOT count the final classifier because it is replaced in a
        # real network. Here, it has an output size of 1000, which is too important
        # and would bias the results.
        # But of course the memory overhead (input + output + weights) is still to be counted in real scenario.
        # With output=2 (binary classification,) and input around 496, the memory overhead is about 2k (so we can neglige it).       
        #activation_hist.append(count_conv_mem(net.classifier))

        if get_hist:
            return max(mem_list), activation_hist

        return max(mem_list)

    def get_efficiency(self, arch_dict):
        self.ofa_net.set_active_subnet(**arch_dict)
        subnet = self.ofa_net.get_active_subnet()
        if torch.cuda.is_available():
            subnet = subnet.cuda()
        data_shape = (1, 3, arch_dict["image_size"], arch_dict["image_size"])
        peak_memory = self.count_peak_activation_size(subnet, data_shape)
        return peak_memory
