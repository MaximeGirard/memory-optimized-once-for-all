# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch.nn as nn
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ofa.utils import AverageMeter, cross_entropy_loss_with_soft_target
from ofa.utils import (
    DistributedMetric,
    list_mean,
    subset_mean,
    val2list,
    MyRandomResizedCrop,
)
from ofa.classification.run_manager import DistributedRunManager

__all__ = [
    "validate",
    "train_one_epoch",
    "train",
    "load_models",
    "train_elastic_depth",
    "train_elastic_expand",
    "train_elastic_width_mult",
]


# Define the validate function
def validate(
    run_manager,
    epoch=0,
    is_test=False,
    image_size_list=None,
    ks_list=None,
    expand_ratio_list=None,
    depth_list=None,
    width_mult_list=None,
    additional_setting=None,
):
    # Get the dynamic network from the run manager
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    # Set the dynamic network to evaluation mode
    dynamic_net.eval()

    # Set default values for the input parameters if not provided
    if image_size_list is None:
        image_size_list = val2list(run_manager.run_config.data_provider.image_size, 1)
    if ks_list is None:
        ks_list = dynamic_net.ks_list
    if expand_ratio_list is None:
        expand_ratio_list = dynamic_net.expand_ratio_list
    if depth_list is None:
        depth_list = dynamic_net.depth_list
    if width_mult_list is None:
        if "width_mult_list" in dynamic_net.__dict__:
            width_mult_list = list(range(len(dynamic_net.width_mult_list)))
        else:
            width_mult_list = [0]

    # Create a list of subnet settings to evaluate
    subnet_settings = []
    for d in depth_list:
        for e in expand_ratio_list:
            for k in ks_list:
                for w in width_mult_list:
                    for img_size in image_size_list:
                        subnet_settings.append(
                            [
                                {
                                    "image_size": img_size,
                                    "d": d,
                                    "e": e,
                                    "ks": k,
                                    "w": w,
                                },
                                "R%s-D%s-E%s-K%s-W%s" % (img_size, d, e, k, w),
                            ]
                        )
    if additional_setting is not None:
        subnet_settings += additional_setting

    # Initialize lists to store losses and accuracies of subnets
    losses_of_subnets, top1_of_subnets, top5_of_subnets = [], [], []

    # Initialize a string to store the validation log
    valid_log = ""

    # Iterate over each subnet setting
    for setting, name in subnet_settings:
        # Write a log for the current subnet setting
        run_manager.write_log(
            "-" * 30 + " Validate %s " % name + "-" * 30, "train", should_print=False
        )
        
        # Assign the active image size to the data provider
        run_manager.run_config.data_provider.assign_active_img_size(
            setting.pop("image_size")
        )
        
        # Set the active subnet in the dynamic network
        dynamic_net.set_active_subnet(**setting)
        
        # Reset the running statistics of the dynamic network
        run_manager.reset_running_statistics(dynamic_net)
        
        # Perform validation on the current subnet setting
        loss, (top1, top5) = run_manager.validate(
            epoch=epoch, is_test=is_test, run_str=name, net=dynamic_net
        )
        
        # Append the loss and accuracies to the respective lists
        losses_of_subnets.append(loss)
        top1_of_subnets.append(top1)
        top5_of_subnets.append(top5)
        
        # Update the validation log
        valid_log += "%s (%.3f), " % (name, top1)

    # Calculate the mean loss and accuracies of all subnets
    return (
        list_mean(losses_of_subnets),
        list_mean(top1_of_subnets),
        list_mean(top5_of_subnets),
        valid_log,
    )


# Define the train_one_epoch function
def train_one_epoch(run_manager, parameters, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.network
    distributed = isinstance(run_manager, DistributedRunManager)

    # switch to train mode
    dynamic_net.train()
    if distributed:
        run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    MyRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    losses = DistributedMetric("train_loss") if distributed else AverageMeter()
    metric_dict = run_manager.get_metric_dict()

    # Use tqdm to display progress bar
    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch + 1),
        disable=distributed and not run_manager.is_root,
    ) as t:
        end = time.time()
        for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
            MyRandomResizedCrop.BATCH = i
            data_time.update(time.time() - end)

            # Adjust learning rate during warmup phase
            if epoch < warmup_epochs:
                new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                    run_manager.optimizer,
                    warmup_epochs * nBatch,
                    nBatch,
                    epoch,
                    i,
                    warmup_lr,
                )
            else:
                new_lr = run_manager.run_config.adjust_learning_rate(
                    run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                )

            images, labels = images.cuda(), labels.cuda()
            target = labels

            # soft target
            if parameters['kd_ratio'] > 0:
                # Set the teacher model to train mode
                parameters['teacher_model'].train()
                # Disable gradient calculation for the teacher model
                with torch.no_grad():
                    # Get the soft logits from the teacher model
                    soft_logits = parameters['teacher_model'](images).detach()
                    # Convert the soft logits to soft labels using softmax function
                    soft_label = F.softmax(soft_logits, dim=1)

            # clean gradients
            dynamic_net.zero_grad()

            loss_of_subnets = []
            subnet_str = ""
            for _ in range(parameters['dynamic_batch_size']):
                # set random seed before sampling
                subnet_seed = int("%d%.3d%.3d" % (epoch * nBatch + i, _, 0))
                random.seed(subnet_seed)
                subnet_settings = dynamic_net.sample_active_subnet()
                subnet_str += (
                    "%d: " % _
                    + ",".join(
                        [
                            "%s_%s"
                            % (
                                key,
                                (
                                    "%.1f" % subset_mean(val, 0)
                                    if isinstance(val, list)
                                    else val
                                ),
                            )
                            for key, val in subnet_settings.items()
                        ]
                    )
                    + " || "
                )

                output = run_manager.net(images)
                if parameters['kd_ratio'] == 0:
                    loss = run_manager.train_criterion(output, labels)
                    loss_type = "ce"
                else:
                    # Calculate the knowledge distillation loss
                    if parameters['kd_type'] == "ce":
                        kd_loss = cross_entropy_loss_with_soft_target(
                            output, soft_label
                        )
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = parameters['kd_ratio'] * kd_loss + run_manager.train_criterion(
                        output, labels
                    )
                    loss_type = "%.1fkd-%s & ce" % (parameters['kd_ratio'], parameters['kd_type'])

                loss_of_subnets.append(loss)
                run_manager.update_metric(metric_dict, output, target)

                loss.backward()
            run_manager.optimizer.step()

            losses.update(list_mean(loss_of_subnets), images.size(0))

            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    **run_manager.get_metric_vals(metric_dict, return_dict=True),
                    "R": images.size(2),
                    "lr": new_lr,
                    "loss_type": loss_type,
                    "seed": str(subnet_seed),
                    "str": subnet_str,
                    "data_time": data_time.avg,
                }
            )
            t.update(1)
            end = time.time()

    return losses.avg.item(), run_manager.get_metric_vals(metric_dict)


def train(run_manager, parameters, validate_func=None):
    # Check if the run_manager is distributed
    distributed = isinstance(run_manager, DistributedRunManager)
    # Set the default validate function to the provided validate_func or the validate function from the previous code
    if validate_func is None:
        validate_func = validate

    # Loop through each epoch
    print("Start training for %d epochs (%d warmups epochs)" % (parameters["n_epochs"], parameters['warmup_epochs']))
    for epoch in range(
        run_manager.start_epoch, parameters["n_epochs"] + parameters['warmup_epochs']
    ):
        # Train one epoch and get the loss and accuracy
        train_loss, (train_top1, train_top5) = train_one_epoch(
            run_manager, parameters, epoch, parameters['warmup_epochs'], parameters['warmup_lr']
        )

        # Check if it's time to perform validation
        if (epoch + 1) % parameters['validation_frequency'] == 0:
            # Perform validation and get the loss, accuracy, and log
            val_loss, val_acc, val_acc5, _val_log = validate_func(
                run_manager, epoch=epoch, is_test=False
            )
            # Check if the current accuracy is the best so far
            is_best = val_acc > run_manager.best_acc
            # Update the best accuracy if necessary
            run_manager.best_acc = max(run_manager.best_acc, val_acc)
            # Check if the run_manager is distributed and if it's the root process
            if not distributed or run_manager.is_root:
                # Create the validation log string
                val_log = (
                    "Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})".format(
                        epoch + 1 - parameters['warmup_epochs'],
                        run_manager.run_config.n_epochs,
                        val_loss,
                        val_acc,
                        run_manager.best_acc,
                    )
                )
                # Add the train top-1 accuracy and train loss to the validation log
                val_log += ", Train top-1 {top1:.3f}, Train loss {loss:.3f}\t".format(
                    top1=train_top1, loss=train_loss
                )
                # Add the validation log from the previous code to the validation log string
                val_log += _val_log
                # Write the validation log to the run_manager's log file
                run_manager.write_log(val_log, "valid", should_print=False)

                # Save the model checkpoint
                run_manager.save_model(
                    {
                        "epoch": epoch,
                        "best_acc": run_manager.best_acc,
                        "optimizer": run_manager.optimizer.state_dict(),
                        "state_dict": run_manager.network.state_dict(),
                    },
                    is_best=is_best,
                )


def load_models(run_manager, dynamic_net, model_path=None):
    # specify init path
    init = torch.load(model_path, map_location="cpu")["state_dict"]
    dynamic_net.load_state_dict(init)
    run_manager.write_log("Loaded init from %s" % model_path, "valid")


def train_elastic_depth(train_func, run_manager, parameters, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    depth_stage_list = dynamic_net.depth_list.copy()
    # We begin to train with the largest values
    depth_stage_list.sort(reverse=True)
    n_stages = len(depth_stage_list) - 1
    current_stage = n_stages - 1
    
    print("- Train with elastic depth using depth_list:", depth_stage_list)
    print("- Current stage: %d/%d" % (current_stage, n_stages - 1))

    # load pretrained models
    if run_manager.start_epoch == 0 and not parameters['resume']:
        validate_func_dict["depth_list"] = sorted(dynamic_net.depth_list)

        load_models(run_manager, dynamic_net, model_path=parameters['ofa_checkpoint_path'])
        # validate after loading weights
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
    else:
        assert parameters['resume']

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Depth: %s -> %s"
        % (depth_stage_list[: current_stage + 1], depth_stage_list[: current_stage + 2])
        + "-" * 30,
        "valid",
    )
    # add depth list constraints
    if (
        len(set(dynamic_net.ks_list)) == 1
        and len(set(dynamic_net.expand_ratio_list)) == 1
    ):
        validate_func_dict["depth_list"] = depth_stage_list
    else:
        validate_func_dict["depth_list"] = sorted(
            {min(depth_stage_list), max(depth_stage_list)}
        )

    # train
    train_func(
        run_manager,
        parameters,
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, **validate_func_dict
        ),
    )


def train_elastic_expand(train_func, run_manager, parameters, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    expand_stage_list = dynamic_net.expand_ratio_list.copy()
    expand_stage_list.sort(reverse=True)
    n_stages = len(expand_stage_list) - 1
    current_stage = n_stages - 1

    print("- Train with elastic expand using expand_ratio_list:", expand_stage_list)
    print("- Current stage: %d/%d" % (current_stage, n_stages - 1))

    # load pretrained models
    if run_manager.start_epoch == 0 and not parameters['resume']:
        validate_func_dict["expand_ratio_list"] = sorted(dynamic_net.expand_ratio_list)

        load_models(run_manager, dynamic_net, model_path=parameters['ofa_checkpoint_path'])
        dynamic_net.re_organize_middle_weights(expand_ratio_stage=current_stage)
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
    else:
        assert parameters['resume']

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Expand Ratio: %s -> %s"
        % (
            expand_stage_list[: current_stage + 1],
            expand_stage_list[: current_stage + 2],
        )
        + "-" * 30,
        "valid",
    )
    if len(set(dynamic_net.ks_list)) == 1 and len(set(dynamic_net.depth_list)) == 1:
        validate_func_dict["expand_ratio_list"] = expand_stage_list
    else:
        validate_func_dict["expand_ratio_list"] = sorted(
            {min(expand_stage_list), max(expand_stage_list)}
        )

    # train
    train_func(
        run_manager,
        parameters,
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, **validate_func_dict
        ),
    )


def train_elastic_width_mult(train_func, run_manager, parameters, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    width_stage_list = dynamic_net.width_mult_list.copy()
    width_stage_list.sort(reverse=True)
    n_stages = len(width_stage_list) - 1
    current_stage = n_stages - 1
    
    print("- Train with elastic width_mult using width_mult_list:", width_stage_list)
    print("- Current stage: %d/%d" % (current_stage, n_stages - 1))

    if run_manager.start_epoch == 0 and not parameters['resume']:
        load_models(run_manager, dynamic_net, model_path=parameters['ofa_checkpoint_path'])
        if current_stage == 0:
            dynamic_net.re_organize_middle_weights(
                expand_ratio_stage=len(dynamic_net.expand_ratio_list) - 1
            )
            run_manager.write_log(
                "reorganize_middle_weights (expand_ratio_stage=%d)"
                % (len(dynamic_net.expand_ratio_list) - 1),
                "valid",
            )
            try:
                dynamic_net.re_organize_outer_weights()
                run_manager.write_log("reorganize_outer_weights", "valid")
            except Exception:
                pass
        run_manager.write_log(
            "%.3f\t%.3f\t%.3f\t%s"
            % validate(run_manager, is_test=True, **validate_func_dict),
            "valid",
        )
    else:
        assert parameters['resume']

    run_manager.write_log(
        "-" * 30
        + "Supporting Elastic Width Mult: %s -> %s"
        % (width_stage_list[: current_stage + 1], width_stage_list[: current_stage + 2])
        + "-" * 30,
        "valid",
    )
    validate_func_dict["width_mult_list"] = sorted({0, len(width_stage_list) - 1})

    # train
    train_func(
        run_manager,
        parameters,
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, **validate_func_dict
        ),
    )
