# MOOFA – a Memory-Optimized OFA architecture for tight memory constraints
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os, os.path as osp

from graphviz import Digraph

k1c = "#%02x%02x%02x" % (0, 128, 0)  # "green"
k3c = "#%02x%02x%02x" % (0, 118, 197)  # "blue"
k5c = "#%02x%02x%02x" % (247, 173, 40)  # yellow
k7c = "#%02x%02x%02x" % (153, 20, 0)  # "red"

c_lut = {1: k1c, 3: k3c, 5: k5c, 7: k7c}

w_lut = {2: 2, 3: 3, 4: 3.5, 6: 4.5}


def get_w(w):
    if w in w_lut:
        return w_lut[w]
    return w + 1


def draw_arch(ofa_net, resolution, out_name="viz/temp", info=None):

    strides = ofa_net.stride_stages

    curr_resolutions = resolution

    ddot = Digraph(
        comment="The visualization of Mojito Architecture Search",
        format="png",
        graph_attr={"ratio": "fill", "size": "10,40"},
        node_attr={"fontsize": "32", "height": "0.8"},
    )
    model_name = "mojito"
    node_cnt = 0
    prev = 0
    with ddot.subgraph(name=model_name) as dot:
        dot.node(
            "%s-%s" % (model_name, node_cnt),
            "Stage 1 (S)",
            fontcolor="black",
            style="rounded,filled",
            shape="record",
            color="lightgray",
            width=str(4.5),
        )
        prev = node_cnt
        node_cnt += 1
        # first conv
        dot.edge(
            "%s-%s" % (model_name, prev),
            "%s-%s" % (model_name, node_cnt),
            fontcolor="black",
            label=f'<<FONT POINT-SIZE="32">{3}x{curr_resolutions}x{curr_resolutions}</FONT>>',
        )
        new_name = f"ConvLayer-{3}x{3}"
        dot.node(
            "%s-%s" % (model_name, node_cnt),
            new_name,
            fontcolor="white",
            style="rounded,filled",
            shape="record",
            color=c_lut[3],
            width=str(get_w(1)),
        )
        prev = node_cnt
        node_cnt += 1
        # stride 2
        curr_resolutions //= 2
        # first mb block
        dot.edge(
            "%s-%s" % (model_name, prev),
            "%s-%s" % (model_name, node_cnt),
            fontcolor="black",
            label=f'<<FONT POINT-SIZE="32">{ofa_net.blocks[0].conv.depth_conv.conv.in_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
        )
        # Residual connection
        dot.edge(
            "%s-%s" % (model_name, prev),
            "%s-%s" % (model_name, node_cnt + 1),
            fontcolor="black",
            label=f'<<FONT POINT-SIZE="32">{ofa_net.blocks[0].conv.depth_conv.conv.in_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
            weight="0",
        )
        new_name = f"MBConv{1}-{3}x{3}"
        dot.node(
            "%s-%s" % (model_name, node_cnt),
            new_name,
            fontcolor="white",
            style="rounded,filled",
            shape="record",
            color=c_lut[3],
            width=str(get_w(1)),
        )

        for stage_id, block_idx in enumerate(ofa_net.block_group_info):
            depth = ofa_net.runtime_depth[stage_id]
            # active_idx allows to select only the x first block of a stage
            active_idx = block_idx[:depth]
            prev = node_cnt
            node_cnt += 1
            if ofa_net.blocks[active_idx[0]].conv.inverted_bottleneck is not None:
                dot.edge(
                    "%s-%s" % (model_name, prev),
                    "%s-%s" % (model_name, node_cnt),
                    fontcolor="black",
                    label=f'<<FONT POINT-SIZE="32">{ofa_net.blocks[active_idx[0]].conv.inverted_bottleneck.conv.conv.in_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
                )
            else:
                dot.edge(
                    "%s-%s" % (model_name, prev),
                    "%s-%s" % (model_name, node_cnt),
                    fontcolor="black",
                    label=f'<<FONT POINT-SIZE="32">{ofa_net.blocks[active_idx[0]].conv.depth_conv.conv.conv.in_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
                )
            dot.node(
                "%s-%s" % (model_name, node_cnt),
                "Stage %d (D)" % (stage_id + 2),
                fontcolor="black",
                style="rounded,filled",
                shape="record",
                color="lightgray",
                width=str(4.5),
            )
            for idx in active_idx:
                if idx == block_idx[1]:
                    curr_resolutions //= strides[stage_id + 1]
                prev = node_cnt
                node_cnt += 1
                if ofa_net.blocks[idx].conv.inverted_bottleneck is not None:
                    dot.edge(
                        "%s-%s" % (model_name, prev),
                        "%s-%s" % (model_name, node_cnt),
                        fontcolor="black",
                        label=f'<<FONT POINT-SIZE="32">{ofa_net.blocks[idx].conv.inverted_bottleneck.conv.conv.in_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
                    )
                else:
                    dot.edge(
                        "%s-%s" % (model_name, prev),
                        "%s-%s" % (model_name, node_cnt),
                        fontcolor="black",
                        label=f'<<FONT POINT-SIZE="32">{ofa_net.blocks[idx].conv.depth_conv.conv.conv.in_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
                    )
                # Residual connection
                if idx in block_idx[1:]:
                    if ofa_net.blocks[idx].conv.inverted_bottleneck is not None:
                        dot.edge(
                            "%s-%s" % (model_name, prev),
                            "%s-%s" % (model_name, node_cnt + 1),
                            fontcolor="black",
                            label=f'<<FONT POINT-SIZE="32">{ofa_net.blocks[idx].conv.inverted_bottleneck.conv.conv.in_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
                            weight="0",
                        )
                    else:
                        dot.edge(
                            "%s-%s" % (model_name, prev),
                            "%s-%s" % (model_name, node_cnt + 1),
                            fontcolor="black",
                            label=f'<<FONT POINT-SIZE="32">{ofa_net.blocks[idx].conv.depth_conv.conv.conv.in_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
                            weight="0",
                        )
                new_name = f"MBConv{ofa_net.blocks[idx].conv.active_expand_ratio}-{ofa_net.blocks[idx].conv.active_kernel_size}x{ofa_net.blocks[idx].conv.active_kernel_size}"
                dot.node(
                    "%s-%s" % (model_name, node_cnt),
                    new_name,
                    fontcolor="white",
                    style="rounded,filled",
                    shape="record",
                    color=c_lut[ofa_net.blocks[idx].conv.active_kernel_size],
                    width=str(get_w(ofa_net.blocks[idx].conv.active_expand_ratio)),
                )

        prev = node_cnt
        node_cnt += 1
        if ofa_net.blocks[-1].conv.inverted_bottleneck is not None:
            dot.edge(
                "%s-%s" % (model_name, prev),
                "%s-%s" % (model_name, node_cnt),
                fontcolor="black",
                label=f'<<FONT POINT-SIZE="32">{ofa_net.blocks[-1].conv.inverted_bottleneck.conv.conv.in_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
            )
        else:
            dot.edge(
                "%s-%s" % (model_name, prev),
                "%s-%s" % (model_name, node_cnt),
                fontcolor="black",
                label=f'<<FONT POINT-SIZE="32">{ofa_net.blocks[-1].conv.depth_conv.conv.conv.in_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
            )
        dot.node(
            "%s-%s" % (model_name, node_cnt),
            "Final stage (S)",
            fontcolor="black",
            style="rounded,filled",
            shape="record",
            color="lightgray",
            width=str(4.5),
        )
        prev = node_cnt
        node_cnt += 1
        dot.edge(
            "%s-%s" % (model_name, prev),
            "%s-%s" % (model_name, node_cnt),
            fontcolor="black",
            label=f'<<FONT POINT-SIZE="32">{ofa_net.final_expand_layer.conv.in_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
        )
        # Final expand layer
        new_name = f"ConvLayer-{1}x{1}"
        dot.node(
            "%s-%s" % (model_name, node_cnt),
            new_name,
            fontcolor="white",
            style="rounded,filled",
            shape="record",
            color=c_lut[1],
            width=str(get_w(1)),
        )
        prev = node_cnt
        node_cnt += 1
        dot.edge(
            "%s-%s" % (model_name, prev),
            "%s-%s" % (model_name, node_cnt),
            fontcolor="black",
            label=f'<<FONT POINT-SIZE="32">{ofa_net.final_expand_layer.conv.out_channels}x{curr_resolutions}x{curr_resolutions}</FONT>>',
        )
        # global averaging
        # Final expand layer
        new_name = f"GlobalAvgPool"
        dot.node(
            "%s-%s" % (model_name, node_cnt),
            new_name,
            fontcolor="white",
            style="rounded,filled",
            shape="record",
            color=c_lut[1],
            width=str(get_w(1)),
        )
        prev = node_cnt
        node_cnt += 1
        dot.edge(
            "%s-%s" % (model_name, prev),
            "%s-%s" % (model_name, node_cnt),
            fontcolor="black",
            label=f'<<FONT POINT-SIZE="32">{ofa_net.feature_mix_layer.conv.in_channels}x{1}x{1}</FONT>>',
        )
        # feature mix layer
        new_name = f"ConvLayer-{1}x{1}"
        dot.node(
            "%s-%s" % (model_name, node_cnt),
            new_name,
            fontcolor="white",
            style="rounded,filled",
            shape="record",
            color=c_lut[1],
            width=str(get_w(1)),
        )
        prev = node_cnt
        node_cnt += 1
        dot.edge(
            "%s-%s" % (model_name, prev),
            "%s-%s" % (model_name, node_cnt),
            fontcolor="black",
            label=f'<<FONT POINT-SIZE="32">{ofa_net.classifier.linear.in_features}x{1}x{1}</FONT>>',
        )
        new_name = f"Linear classifier"
        dot.node(
            "%s-%s" % (model_name, node_cnt),
            new_name,
            fontcolor="white",
            style="rounded,filled",
            shape="record",
            color=c_lut[1],
            width=str(get_w(1)),
        )

    if info is not None:
        res = []
        for k, v in info.items():
            res.append("%s: %.2f" % (k, v))
        result = " ".join(res)
        ddot.attr(label=f'<<FONT POINT-SIZE="32">{result}</FONT>>', labelloc="top")

    os.makedirs(osp.dirname(out_name), exist_ok=True)
    ddot.render(out_name)
    # print(f"The arch is visualized to {out_name}")
