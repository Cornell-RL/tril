from accelerate import Accelerator
from accelerate.utils import (
    DeepSpeedEngineWrapper,
    DeepSpeedOptimizerWrapper,
    DeepSpeedSchedulerWrapper,
    DummyOptim,
    DummyScheduler,
    DistributedType,
)

import torch
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from tril.agent import Agent

class DeepspeedMultiEngineAccelerator(Accelerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deepspeed_engine = {}

    def prepare(self, *args, device_placement=None):
        if device_placement is None:
            device_placement = [None for _ in args]
        elif self.distributed_type in (DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM):
            raise ValueError("You can't customize device placements with DeepSpeed or Megatron-LM.")
        elif len(device_placement) != len(args):
            raise ValueError(
                f"`device_placement` should be a list with {len(args)} elements (the number of objects passed)."
            )

        for obj in args:
            # TODO: Look at enabling native TP training directly with a proper config
            if (
                isinstance(obj, torch.nn.Module)
                and self.verify_device_map(obj)
                and self.distributed_type != DistributedType.NO
                and os.environ.get("ACCELERATE_BYPASS_DEVICE_MAP", "false") != "true"
            ):
                raise ValueError(
                    "You can't train a model that has been loaded with `device_map='auto'` in any distributed mode."
                    " Please rerun your script specifying `--num_processes=1` or by launching with `python {{myscript.py}}`."
                )

        if self.distributed_type == DistributedType.FSDP:
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

            model_count = 0
            optimizer_present = False
            is_type_fsdp = False
            for obj in args:
                if isinstance(obj, torch.nn.Module):
                    model_count += 1
                    # if the model is compiled using PyTorch 2.0,
                    # check that the wrapped model is FSDP or not;
                    # else check if it is FSDP or not;
                    is_type_fsdp = isinstance(obj, FSDP) or (
                        is_compiled_module(obj) and isinstance(obj._orig_mod, FSDP)
                    )
                if isinstance(obj, torch.optim.Optimizer):
                    optimizer_present = True
            if model_count > 1 and optimizer_present:
                raise ValueError(
                    "For FSDP to work with multiple models (>1), "
                    "prepare must be called for all the models before optimizers are created. "
                    "Then pass the optimizers to the prepare call in the same order as corresponding models."
                )
            elif model_count == 1 and not is_type_fsdp and optimizer_present:
                logger.warning(
                    "FSDP Warning: When using FSDP, "
                    "it is efficient and recommended to call prepare for the model before creating the optimizer"
                )

        if self.distributed_type == DistributedType.DEEPSPEED:
            model_count = 0
            for obj in args:
                if isinstance(obj, torch.nn.Module):
                    model_count += 1
            #if model_count > 1:
            #    raise AssertionError(
            #        "You can't use same `Accelerator()` instance with multiple models when using DeepSpeed"
            #    )

        # On TPUs, putting the model on the XLA device will create new parameters, so the corresponding optimizer will
        # have parameters disconnected from the model (so no training :-( ).
        # If the model and optimizer have parameters on different devices we raise an error.
        if self.distributed_type == DistributedType.TPU:
            model_device, optimizer_device = self._get_devices()
            if model_device is not None and optimizer_device is not None and model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you "
                    "created an optimizer around your model **before** putting on the device. Make sure the line "
                    "model.to(device) is before the optimizer creation in your script or remove it entirely and use "
                    "the flag default value for `device_placement` in your `Accelerator` to let it handle that "
                    "part for you."
                )

        # If we're dealing with device placement, this deals with that by...
        tpu_should_fix_optimizer = self.device_placement and self.distributed_type == DistributedType.TPU
        if tpu_should_fix_optimizer or self.mixed_precision == "fp8":
            # 1. grabbing old model parameters
            old_named_params = self._get_named_parameters(*args)

        if self.distributed_type in [DistributedType.MULTI_CPU, DistributedType.MULTI_XPU, DistributedType.NO]:
            if self.device.type == "cpu" and self.state.use_ipex:
                args = self._prepare_ipex(*args)
            elif self.device.type == "xpu" and is_xpu_available():
                args = self._prepare_ipex(*args)
        if self.distributed_type == DistributedType.DEEPSPEED:
            result = self._prepare_deepspeed(*args)
        elif self.distributed_type == DistributedType.MEGATRON_LM:
            result = self._prepare_megatron_lm(*args)
        else:
            result = tuple(
                self._prepare_one(obj, first_pass=True, device_placement=d) for obj, d in zip(args, device_placement)
            )
            result = tuple(self._prepare_one(obj, device_placement=d) for obj, d in zip(result, device_placement))

        if tpu_should_fix_optimizer or self.mixed_precision == "fp8":
            # 2. grabbing new model parameters
            new_named_params = self._get_named_parameters(*result)
            # 3. building a map from the first to the second
            mapping = {p: new_named_params[n] for n, p in old_named_params.items()}
            # 4. using that map to update the parameters of the optimizer
            for obj in result:
                if isinstance(obj, torch.optim.Optimizer):
                    obj._switch_parameters(mapping)

        if (
            self.distributed_type == DistributedType.FSDP
            and model_count == 1
            and not is_type_fsdp
            and optimizer_present
        ):
            result = self._prepare_fsdp(*result)

        for item in result:
            if any(
                item in container
                for container in (self._dataloaders, self._models, self._optimizers, self._schedulers)
            ):
                setattr(item, "_is_accelerate_prepared", True)

        return result if len(result) > 1 else result[0]

    def _prepare_deepspeed(self, *args):
        import deepspeed

        deepspeed_plugin = self.state.deepspeed_plugin

        is_dataloader_present = any(isinstance(obj, torch.utils.data.DataLoader) for obj in args)
        if deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto" or is_dataloader_present:
            result = [
                self._prepare_one(obj, first_pass=True) if isinstance(obj, torch.utils.data.DataLoader) else obj
                for obj in args
            ]

            batch_sizes = [obj.batch_size for obj in args if hasattr(obj, "batch_size")]
            if self.split_batches:
                batch_sizes = [batch_size // self.num_processes for batch_size in batch_sizes]

            if any(bs is None for bs in batch_sizes):
                raise ValueError(
                    "At least one of the dataloaders passed to `accelerate.prepare()` has `None` as batch size."
                    "Please set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file"
                    "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                )
            if len(batch_sizes) == 0:
                raise ValueError(
                    "When using DeepSpeed `accelerate.prepare()` requires you to pass at least one of training or evaluation dataloaders "
                    "or alternatively set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file"
                    "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                )

            batch_size_per_device = min(batch_sizes) if deepspeed_plugin.is_train_batch_min else max(batch_sizes)
            #if len(batch_sizes) > 1:
            #    logger.info(
            #        "Since you passed both train and evaluation dataloader, `is_train_batch_min` (here "
            #        f"{deepspeed_plugin.is_train_batch_min} will decide the `train_batch_size` ({batch_size_per_device})."
            #    )
        else:
            batch_size_per_device = deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"]
            result = [obj for obj in args]

        # handle `gradient_accumulation_steps` when the value is `auto`
        deepspeed_plugin.fill_match(
            "gradient_accumulation_steps",
            must_match=False,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        config_kwargs = {
            "train_micro_batch_size_per_gpu": batch_size_per_device,
            "train_batch_size": batch_size_per_device
            * deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]
            * self.num_processes,
            "gradient_clipping": 1.0,
            "zero_optimization.stage3_gather_16bit_weights_on_model_save": False,
        }

        # JONATHAN: going to create multiple optimizers and schedulers
        # TODO: perhaps this is how we can get multiple models as well with deepspeed...

        # 1: get the agent

        # 2: check if reward is trainable
        #if True:
        if len(result) < 2 or not result[1].is_trainable:  # <- this means only 1 engine

            model = None
            optimizer = None
            scheduler = None
            for obj in result:
                if isinstance(obj, torch.nn.Module):
                    model = obj
                elif isinstance(obj, (torch.optim.Optimizer, DummyOptim)):
                    optimizer = obj
                    #optimizer = [obj] if optimizer is None else optimizer.append(obj)
                elif (isinstance(obj, (LRScheduler, DummyScheduler))) or (
                    type(obj).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                ):
                    scheduler = obj
                    #import pdb; pdb.set_trace()
                    #scheduler = [obj] if scheduler is None else scheduler.append(obj)

            if optimizer is not None:
                #for optim in optimizer:
                if "optimizer" in deepspeed_plugin.deepspeed_config and not isinstance(optimizer, (DummyOptim)):
                #if "optimizer" in deepspeed_plugin.deepspeed_config and not isinstance(optim, (DummyOptim)):
                    raise ValueError(
                        "You cannot specify an optimizer in the config file and in the code at the same time. "
                        "Please remove the optimizer from the config file or "
                        "create `accelerate.utils.DummyOptim` in the code."
                    )
                elif "optimizer" not in deepspeed_plugin.deepspeed_config and isinstance(optimizer, (DummyOptim)):
                #elif "optimizer" not in deepspeed_plugin.deepspeed_config and isinstance(optim, (DummyOptim)):
                    raise ValueError(
                        "You cannot create a `DummyOptim` without specifying an optimizer in the config file."
                    )

                if isinstance(optimizer, (torch.optim.Optimizer)):
                #if isinstance(optim, (torch.optim.Optimizer)):
                    deepspeed_plugin.deepspeed_config["zero_allow_untested_optimizer"] = True

            if scheduler is not None:
                #for sch in scheduler:
                if "scheduler" in deepspeed_plugin.deepspeed_config and not isinstance(scheduler, (DummyScheduler)):
                #if "scheduler" in deepspeed_plugin.deepspeed_config and not isinstance(sch, (DummyScheduler)):
                    raise ValueError(
                        "You cannot specify a scheduler in the config file and in the code at the same time. "
                        "Please remove the scheduler from the config file or "
                        "create `accelerate.utils.DummyScheduler` in the code."
                    )
                elif (
                    "scheduler" not in deepspeed_plugin.deepspeed_config
                    and isinstance(scheduler, (DummyScheduler))
                    and scheduler.lr_scheduler_callable is None
                    #and isinstance(sch, (DummyScheduler))
                    #and sch.lr_scheduler_callable is None
                ):
                    raise ValueError(
                        "Either specify a scheduler in the config file or "
                        "pass in the `lr_scheduler_callable` parameter when using `accelerate.utils.DummyScheduler`."
                    )

            # TODO: come up with better way to group optimizers, schedulers, per engine
            #if optimizer is not None and scheduler is not None:
            #    if isinstance(optimizer, (DummyOptim)) and not isinstance(scheduler, (DummyScheduler)):
            #        raise ValueError(
            #            "You can only specify `accelerate.utils.DummyScheduler` in the code when using "
            #            "`accelerate.utils.DummyOptim`."
            #        )

            if model is not None:
                if hasattr(model, "config"):
                    hidden_size = (
                        max(model.config.hidden_sizes)
                        if getattr(model.config, "hidden_sizes", None)
                        else getattr(model.config, "hidden_size", None)
                    )
                    if hidden_size is not None:
                        config_kwargs.update(
                            {
                                "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                                "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                                "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            }
                        )

                # NOTE: at this point just assuming we're not using DummyOptim or DummyScheduler
                #if isinstance(optimizer, (DummyOptim)):
                #    config_kwargs.update(
                #        {"optimizer.params.lr": optimizer.lr, "optimizer.params.weight_decay": optimizer.weight_decay}
                #    )
                #if isinstance(scheduler, (DummyScheduler)) and scheduler.lr_scheduler_callable is None:
                #    max_lr = (
                #        getattr(scheduler.optimizer, "lr", None)
                #        if getattr(scheduler.optimizer, "defaults", None) is None
                #        else scheduler.optimizer.defaults["lr"]
                #    )
                #    config_kwargs.update(
                #        {
                #            "scheduler.params.warmup_min_lr": 0,
                #            "scheduler.params.warmup_max_lr": max_lr,
                #            "scheduler.params.warmup_num_steps": scheduler.warmup_num_steps,
                #        }
                #    )
                #    if scheduler.total_num_steps is not None:
                #        config_kwargs["scheduler.params.total_num_steps"] = (
                #            math.ceil(scheduler.total_num_steps / self.num_processes)
                #            if not self.split_batches
                #            else scheduler.total_num_steps
                #        )
                deepspeed_plugin.deepspeed_config_process(must_match=False, **config_kwargs)
                self.deepspeed_config = deepspeed_plugin.deepspeed_config
                kwargs = dict(model=model, config_params=self.deepspeed_config)
                if optimizer is not None:
                    if isinstance(optimizer, (DummyOptim)):
                        kwargs["model_parameters"] = optimizer.params
                        if isinstance(scheduler, (DummyScheduler)) and scheduler.lr_scheduler_callable is not None:
                            kwargs["lr_scheduler"] = scheduler.lr_scheduler_callable
                    else:
                        if self.deepspeed_config["zero_optimization"].get("offload_optimizer", {}).get(
                            "device", "none"
                        ) != "none" and self.deepspeed_config.get("zero_force_ds_cpu_optimizer", True):
                            from deepspeed.ops.adam import DeepSpeedCPUAdam

                            defaults = {k: v for k, v in optimizer.defaults.items() if k in ["lr", "weight_decay"]}
                            optimizer = DeepSpeedCPUAdam(optimizer.param_groups, **defaults)
                        kwargs["optimizer"] = optimizer
                        if scheduler is not None:
                            if (
                                isinstance(scheduler, LRScheduler)
                                or type(scheduler).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                            ):
                                kwargs["lr_scheduler"] = scheduler
                engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)
                if optimizer is not None:
                    optimizer = DeepSpeedOptimizerWrapper(optimizer)
                if scheduler is not None:
                    if lr_scheduler is None:
                        scheduler = AcceleratedScheduler(
                            scheduler,
                            optimizer,
                            step_with_optimizer=self.step_scheduler_with_optimizer,
                            split_batches=self.split_batches,
                        )
                    else:
                        scheduler = DeepSpeedSchedulerWrapper(lr_scheduler, optimizer)

                for i in range(len(result)):
                    if isinstance(result[i], torch.nn.Module):
                        result[i] = engine
                    elif isinstance(result[i], (torch.optim.Optimizer, DummyOptim)):
                        result[i] = optimizer
                    elif (isinstance(result[i], (LRScheduler, DummyScheduler))) or (
                        type(result[i]).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                    ):
                        result[i] = scheduler
                # pointing for deepspeed_engine_wrapped.backward()
                #self.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)
                self.deepspeed_engine['policy'] = DeepSpeedEngineWrapper(engine)
                self._models.append(engine)
                if optimizer is not None:
                    self._optimizers.append(optimizer)
                if scheduler is not None:
                    self._schedulers.append(scheduler)
                if len(self._models) > 1:
                    raise AssertionError(
                        "You can't use same `Accelerator()` instance with multiple models when using DeepSpeed"
                    )
            return tuple(result)
        else:
            models = [result[0], result[1]]
            optimizers = []
            scheduler = None
            for obj in result:
                if isinstance(obj, (torch.optim.Optimizer, DummyOptim)):
                    optimizers.append(obj)
                elif (isinstance(obj, (LRScheduler, DummyScheduler))) or (
                    type(obj).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                ):
                    #scheduler = [obj] if scheduler is None else scheduler.append(obj)
                    scheduler = obj

            for idx, (model, optimizer, label) in enumerate(zip(models, optimizers, ['policy', 'reward'])):
                if hasattr(model, "config"):
                    hidden_size = (
                        max(model.config.hidden_sizes)
                        if getattr(model.config, "hidden_sizes", None)
                        else getattr(model.config, "hidden_size", None)
                    )
                    if hidden_size is not None:
                        config_kwargs.update(
                            {
                                "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                                "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                                "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            }
                        )

                deepspeed_plugin.deepspeed_config_process(must_match=False, **config_kwargs)
                self.deepspeed_config = deepspeed_plugin.deepspeed_config
                kwargs = dict(model=model, config_params=self.deepspeed_config)
                if optimizer is not None:
                    if self.deepspeed_config["zero_optimization"].get("offload_optimizer", {}).get(
                        "device", "none"
                    ) != "none" and self.deepspeed_config.get("zero_force_ds_cpu_optimizer", True):
                        from deepspeed.ops.adam import DeepSpeedCPUAdam

                        defaults = {k: v for k, v in optimizer.defaults.items() if k in ["lr", "weight_decay"]}
                        optimizer = DeepSpeedCPUAdam(optimizer.param_groups, **defaults)
                        if scheduler is not None and idx == 0:
                            if (
                                isinstance(scheduler, LRScheduler)
                                or type(scheduler).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                            ):
                                kwargs["lr_scheduler"] = scheduler
                    kwargs["optimizer"] = optimizer
                engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)
                if optimizer is not None:
                    optimizer = DeepSpeedOptimizerWrapper(optimizer)
                if scheduler is not None and idx == 0:
                    if lr_scheduler is None:
                        scheduler = AcceleratedScheduler(
                            scheduler,
                            optimizer,
                            step_with_optimizer=self.step_scheduler_with_optimizer,
                            split_batches=self.split_batches,
                        )
                    else:
                        scheduler = DeepSpeedSchedulerWrapper(lr_scheduler, optimizer)

                #for i in range(len(result)):
                #    if isinstance(result[i], torch.nn.Module):
                #        result[i] = engine
                #    elif isinstance(result[i], (torch.optim.Optimizer, DummyOptim)):
                #        result[i] = optimizer
                #    elif (isinstance(result[i], (LRScheduler, DummyScheduler))) or (
                #        type(result[i]).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                #    ):
                #        result[i] = scheduler

                result[idx] = engine
                result[idx+2] = optimizer
                # pointing for deepspeed_engine_wrapped.backward()
                #self.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)
                self.deepspeed_engine[label] = DeepSpeedEngineWrapper(engine)
                self._models.append(engine)
                if optimizer is not None:
                    self._optimizers.append(optimizer)
                if scheduler is not None and idx == 0:
                    self._schedulers.append(scheduler)
                #if len(self._models) > 1:
                #    raise AssertionError(
                #        "You can't use same `Accelerator()` instance with multiple models when using DeepSpeed"
                #    )
            for i in range(len(result)):
                if (isinstance(result[i], (LRScheduler, DummyScheduler))) or (
                    type(result[i]).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                ):
                    result[i] = scheduler
            return tuple(result)

    def backward(self, loss, **kwargs):
        """
        Scales the gradients in accordance to the `GradientAccumulationPlugin` and calls the correct `backward()` based
        on the configuration.

        Should be used in lieu of `loss.backward()`.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=2)
        >>> outputs = model(inputs)
        >>> loss = loss_fn(outputs, labels)
        >>> accelerator.backward(loss)
        ```
        """

        if self.distributed_type != DistributedType.DEEPSPEED:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.gradient_accumulation_steps
        if self.distributed_type == DistributedType.DEEPSPEED:
            #self.deepspeed_engine_wrapped.backward(loss, **kwargs)
            engine_type = kwargs.pop("model_fn", "policy")
            self.deepspeed_engine[engine_type].backward(loss, **kwargs)
        elif self.distributed_type == DistributedType.MEGATRON_LM:
            return
        elif self.scaler is not None:
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)


if __name__ == '__main__':
    accelerator = DeepspeedMultiEngineAccelerator()
    accelerator._prepare_deepspeed()

