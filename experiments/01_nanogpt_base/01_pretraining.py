"""
This is a placeholder, real implementation will have logging, checkpointing
And proper usage of 2 different optimizers
"""

import torch

from research_lib.architectures.config import NanoGPTConfig
from research_lib.architectures.modded_nanogpt_base import ModdedNanoGPT

if __name__ == "__main__":
    config = NanoGPTConfig(...)
    model = ModdedNanoGPT(config).cuda()

    # Compile for training
    model = torch.compile(model, mode="reduce-overhead")

    # Train!
    optimizer = torch.optim.AdamW(model.parameters())
    for batch in dataloader:
        logits = model(batch["input_ids"])
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size), batch["labels"].view(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
