#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import time
from functools import partial

import datasets
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
import transformers
from transformers import CONFIG_MAPPING, MODEL_MAPPING, AutoConfig, LlamaForCausalLM
from transformers.utils.versions import require_version

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.tensor import ProcessGroup, ShardSpec
from colossalai.utils import get_current_device
from colossalai.zero import ColoInitContext, GeminiAdamOptimizer, GeminiDDP


def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_time_stamp():
    torch.cuda.synchronize()
    return time.time()


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per dp group) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20,
        help="Total number of training steps to perform.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--shardinit",
        action="store_true",
        help="Initialize the model with tensor parallel",
    )
    parser.add_argument("--mem_cap", type=int, default=0, help="use mem cap")
    parser.add_argument("--init_in_cpu", action='store_true', default=False, help="init training model in cpu")
    args = parser.parse_args()

    return args


def colo_memory_cap(size_in_GB):
    from colossalai.utils import colo_device_memory_capacity, colo_set_process_memory_fraction, get_current_device
    cuda_capacity = colo_device_memory_capacity(get_current_device())
    if size_in_GB * (1024**3) < cuda_capacity:
        colo_set_process_memory_fraction(size_in_GB * (1024**3) / cuda_capacity)
        print("Using {} GB of GPU memory".format(size_in_GB))


def main():
    args = parse_args()
    SEQ_LEN = 1024
    VOCAB_SIZE = 32000
    input_ids, attn_mask = get_data(args.batch_size, SEQ_LEN, VOCAB_SIZE)

    config = {}
    config['parallel'] = {
        'pipeline': 1,
        'tensor': dict(size=2, mode='1d'),
    }

    disable_existing_loggers()
    colossalai.launch_from_torch(config)
    logger = get_dist_logger()
    is_main_process = dist.get_rank() == 0

    if is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.mem_cap > 0:
        colo_memory_cap(args.mem_cap)

    # If passed along, set the training seed now.
    if args.seed is not None:
        torch.mannul_seed(args.seed)
        logger.info(f"Rank {dist.get_rank()}: random seed is set to {args.seed}")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    logger.info("Model config has been created", ranks=[0])
    #config = {
    #    "_name_or_path": "decapoda-research/llama-7b-hf",
    #    "architectures": [
    #      "LLaMAForCausalLM"
    #    ],
    #    "bos_token_id": 0,
    #    "eos_token_id": 1,
    #    "hidden_act": "silu",
    #    "hidden_size": 4096,
    #    "initializer_range": 0.02,
    #    "intermediate_size": 11008,
    #    "max_position_embeddings": 2048,
    #    "max_sequence_length": 2048,
    #    "model_type": "llama",
    #    "num_attention_heads": 32,
    #    "num_hidden_layers": 32,
    #    "pad_token_id": -1,
    #    "rms_norm_eps": 1e-06,
    #    "tie_word_embeddings": False,
    #    "torch_dtype": "float16",
    #    "transformers_version": "4.28.1",
    #    "use_cache": True,
    #    "vocab_size": 32000
    #}

    if args.init_in_cpu:
        init_dev = torch.device('cpu')
    else:
        init_dev = get_current_device()

    # shard init prameters
    if args.shardinit:
        logger.info("Sharding initialization !", ranks=[0])
    else:
        logger.info("Skipping sharding initialization", ranks=[0])

    world_size = torch.distributed.get_world_size()
    shard_pg = ProcessGroup(tp_degree=world_size) if args.shardinit else None
    default_dist_spec = ShardSpec([-1], [world_size]) if args.shardinit else None

    # build model
    if args.model_name_or_path is None:
        logger.info("Train a new model from scratch", ranks=[0])
        with ColoInitContext(device=init_dev,
                             dtype=torch.half,
                             default_dist_spec=default_dist_spec,
                             default_pg=shard_pg):
            model = LlamaForCausalLM(config)
    else:
        logger.info("Finetune a pre-trained model", ranks=[0])
        with ColoInitContext(device=init_dev,
                             dtype=torch.half,
                             default_dist_spec=default_dist_spec,
                             default_pg=shard_pg):
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,
                                                   from_tf=bool(".ckpt" in args.model_name_or_path),
                                                   config=config,
                                                   local_files_only=False)

    # enable graident checkpointing
    model.gradient_checkpointing_enable()


    numel = sum([p.numel() for p in model.parameters()])
    model = GeminiDDP(model, device=get_current_device(), placement_policy="cpu", pin_memory=True, strict_ddp_mode=args.shardinit)
    optimizer = GeminiAdamOptimizer(model, lr=args.learning_rate, initial_scale=2**14, gpu_margin_mem_ratio=0.0)

    get_tflops_func = partial(get_tflops, numel, args.batch_size, SEQ_LEN)

    model.train()
    
    def train_step():
        input_ids, attn_mask = get_data(args.batch_size, SEQ_LEN, VOCAB_SIZE)
        st_time = time.time()
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids, use_cache=False)
        loss = outputs['loss']
        fwd_end = time.time()
        fwd_time = fwd_end - st_time
        step_tflops = get_tflops_func(fwd_time)
        logger.info("forward finished", ranks=[0])

        optimizer.backward(loss)
        torch.cuda.synchronize()
        bwd_end = time.time()
        bwd_time = bwd_end - fwd_end
        logger.info("backward finished", ranks=[0])

        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        optim_time = time.time() - bwd_end
        step_time = time.time() - st_time
        logger.info("optimizer finished", ranks=[0])

        step_tflops = get_tflops_func(step_time)
        logger.info(
            f"[Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s",
            ranks=[0],
        )
    
    warmup_steps = 5
    active_steps = 1
    save_dir = "./profiling"
    demo_profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(wait=0, warmup=warmup_steps, active=active_steps),
                       on_trace_ready=tensorboard_trace_handler(save_dir),
                       record_shapes=True,
                       profile_memory=True)
    with demo_profiler as prof:
        for n in range(args.max_train_steps):
            train_step()
            prof.step() 
    
    
    # for step in range(args.max_train_steps):
    #     st_time = time.time()
    #     input_ids, attn_mask = get_data(args.batch_size, SEQ_LEN, VOCAB_SIZE)
    #     outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids, use_cache=False)
    #     loss = outputs['loss']
    #     optimizer.backward(loss)

    #     optimizer.step()
    #     optimizer.zero_grad()
    #     torch.cuda.synchronize()
    #     step_time = time.time() - st_time
    #     step_tflops = get_tflops_func(step_time)

    #     logger.info("step {} finished, Tflops {}".format(step, step_tflops), ranks=[0])

    logger.info("Training finished", ranks=[0])


if __name__ == "__main__":
    main()

