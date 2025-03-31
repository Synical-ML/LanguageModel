import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset
import socket

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def load_json_config(filename):
    with open(filename, 'r') as f:
        return json.load(f)


model_config = load_json_config('model_config.json')
dataset_config = load_json_config('dataset_config.json')
train_config = load_json_config('train_config.json')


from model_utils import GPTConfig, GPTModel


def setup_distributed(main_process_ip='localhost', main_process_port=29500):
    os.environ['MASTER_ADDR'] = main_process_ip
    os.environ['MASTER_PORT'] = str(main_process_port)

    dist.init_process_group(backend='nccl')

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    return local_rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def prepare_dataset(dataset_config, world_size=1, rank=0):

    dataset = load_dataset(
        dataset_config['dataset_name'],
        dataset_config.get('dataset_subset', None),
        split=dataset_config['dataset_split'],
        streaming=dataset_config.get('stream_dataset', False)
    )

    if not dataset_config.get('stream_dataset', False):
        dataset = dataset.shard(num_shards=world_size, index=rank)

    return dataset


class DistributedIterableDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length, dataset_key, batch_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_key = dataset_key
        self.batch_size = batch_size

    def __iter__(self):
        batch_input_ids = []
        batch_attention_masks = []

        for item in self.dataset:
            text = item[self.dataset_key]

            encodings = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.max_length+1,
                return_tensors='pt'
            )

            input_ids = encodings['input_ids'].squeeze()
            attention_mask = encodings['attention_mask'].squeeze()

            if input_ids.numel() < 2:
                continue

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)

            if len(batch_input_ids) == self.batch_size:
                batch_tensor = self.pad_batch(batch_input_ids, batch_attention_masks)

                yield batch_tensor

                batch_input_ids = []
                batch_attention_masks = []

        if batch_input_ids:
            batch_tensor = self.pad_batch(batch_input_ids, batch_attention_masks)
            yield batch_tensor

    def pad_batch(self, input_ids_list, attention_masks_list):
        max_len = max(ids.size(0) for ids in input_ids_list)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids_list, attention_masks_list):
            pad_length = max_len - ids.size(0)
            padded_ids = torch.nn.functional.pad(
                ids,
                (0, pad_length),
                value=self.tokenizer.pad_token_id
            )

            padded_mask = torch.nn.functional.pad(
                mask,
                (0, pad_length),
                value=0
            )

            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks)
        }


def train():
    main_process_ip = train_config.get('main_process_ip', 'localhost')
    main_process_port = train_config.get('main_process_port', 29500)

    try:
        local_rank, world_size = setup_distributed(main_process_ip, main_process_port)
    except Exception as e:
        print(f"Distributed setup failed: {e}")
        return

    use_wandb = train_config.get('use_wandb', False) and WANDB_AVAILABLE and local_rank == 0

    tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer'])
    tokenizer.pad_token_id = tokenizer.eos_token_id

    config = GPTConfig(
        max_length=model_config['max_length'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_model=model_config['d_model'],
        rate=model_config['rate'],
        tokenizer=tokenizer
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPTModel(config).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    base_dataset = prepare_dataset(dataset_config, world_size, local_rank)

    dataset = DistributedIterableDataset(
        base_dataset,
        tokenizer,
        model_config['max_length'],
        dataset_config['dataset_key'],
        train_config["batch_size"] // world_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=4,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )

    if use_wandb:
        wandb.init(
            project=f"LLM",
            config={
                **model_config,
                **dataset_config,
                **train_config,
                "world_size": world_size,
                "distributed": True
            }
        )
        wandb.watch(model, log="all", log_freq=10)

    model.train()
    for epoch in range(train_config['epochs']):
        total_loss = 0
        epoch_loss = 0
        batch_count = 0

        if local_rank == 0:
            if TQDM_AVAILABLE and not use_wandb:
                progress_bar = tqdm(desc=f"Epoch {epoch + 1}", unit="batch")

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)#.cuda(non_blocking=True)
            attention_mask = batch['attention_mask'].to(device)#.cuda(non_blocking=True)

            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

            logits = model(input_ids)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()

            reduced_loss = loss.clone()
            dist.all_reduce(reduced_loss)
            reduced_loss /= world_size

            total_loss += reduced_loss.item()
            epoch_loss += reduced_loss.item()
            batch_count += 1

            if local_rank == 0:
                if use_wandb:
                    wandb.log({
                        "batch_loss": reduced_loss.item(),
                        "epoch": epoch,
                    })
                elif TQDM_AVAILABLE:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": reduced_loss.item()})
                else:
                    print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {reduced_loss.item()}")

                if batch_idx % train_config.get('save_interval', 1000) == 0:
                    torch.save(model.state_dict(), train_config['save_path'])

            if batch_count >= train_config.get('max_batches_per_epoch', float('inf')):
                break

        if local_rank == 0:
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            if use_wandb:
                wandb.log({
                    "epoch_loss": avg_epoch_loss,
                    "epoch": epoch
                })
            elif TQDM_AVAILABLE:
                progress_bar.close()

    if local_rank == 0:
        if use_wandb:
            wandb.finish()

    cleanup_distributed()

    if local_rank == 0:
        print("Training completed!")


if __name__ == "__main__":
    train()
