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
            # we assume we only need to store input and output, the weights are partially loaded for computation
            if m is None:
                return 0
            if hasattr(m, "conv"):
                m = m.conv
            elif hasattr(m, "linear"):
                m = m.linear
            assert isinstance(m, (nn.Conv2d, nn.Linear))
            return m.input_size.item() + m.output_size.item()

        def count_block(m, get_list=False):
            assert isinstance(m, ResidualBlock)

            if m.conv is None or isinstance(
                m.conv, ZeroLayer
            ):  # just an identical mapping
                return 0

            assert isinstance(m.conv, MBConvLayer)

            if m.shortcut is None or isinstance(
                m.shortcut, ZeroLayer
            ):  # no residual connection, just convs
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
            else:  # convs and residual
                if m.conv.inverted_bottleneck is not None:
                    residual_size = m.conv.inverted_bottleneck.conv.input_size.item()
                else:
                    residual_size = m.conv.depth_conv.conv.input_size.item()

                # consider residual size for later layers
                if get_list:
                    return [
                        count_conv_mem(m.conv.inverted_bottleneck),
                        count_conv_mem(m.conv.depth_conv) + residual_size,
                        # | The output of point_linear conv is directly added to the residual in the forward pass
                        # v No memory overload is needed to store output then perform addition
                        count_conv_mem(m.mobile_inverted_conv.point_linear),
                    ]
                else:
                    return max(
                        [
                            count_conv_mem(m.conv.inverted_bottleneck),
                            count_conv_mem(m.conv.depth_conv) + residual_size,
                            # | The output of point_linear conv is directly added to the residual in the forward pass
                            # v No memory overload is needed to store output then perform addition
                            count_conv_mem(m.mobile_inverted_conv.point_linear),
                        ]
                    )

        if isinstance(net, nn.DataParallel):
            net = net.module
        net = copy.deepcopy(net)

        assert isinstance(net, MobileNetV3)

        # record the input and output size
        net.apply(add_io_hooks)
        # run a dummy input to record the size
        with torch.no_grad():
            _ = net(torch.randn(*data_shape).to(net.parameters().__next__().device))

        mem_list = [
            count_conv_mem(net.first_conv),
            count_conv_mem(net.final_expand_layer),
            count_conv_mem(net.feature_mix_layer),
            count_conv_mem(net.classifier),
        ] + [count_block(blk) for blk in net.blocks]

        activation_hist.append(count_conv_mem(net.first_conv))
        for blk in net.blocks:
            activation_hist += [count for count in count_block(blk, get_list=True) if count != 0]
        activation_hist.append(count_conv_mem(net.final_expand_layer))
        activation_hist.append(count_conv_mem(net.feature_mix_layer))
        activation_hist.append(count_conv_mem(net.classifier))
        
        if get_hist:
            return max(mem_list), activation_hist

        return max(mem_list)  # pick the peak mem

    # Return the number of parameter of the network during a memory peak
    # in bytes if each parameter is stored on 8 bits
    def get_efficiency(self, arch_dict):
        self.ofa_net.set_active_subnet(**arch_dict)
        subnet = self.ofa_net.get_active_subnet()
        if torch.cuda.is_available():
            subnet = subnet.cuda()
        data_shape = (1, 3, arch_dict["image_size"], arch_dict["image_size"])
        peak_memory = self.count_peak_activation_size(subnet, data_shape)
        # print(peak_memory)
        return peak_memory
