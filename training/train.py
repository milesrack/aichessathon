"""Train the original residual evaluator from a prepared, hash-split dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Evaluator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(769, 32, padding_idx=768)
        self.output = nn.Linear(32, 1)
        nn.init.normal_(self.embedding.weight, std=0.03)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.embedding.weight[768].zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.embedding(features).sum(2))
        scores = self.output(hidden).squeeze(-1)
        return 200 * torch.tanh((scores[:, 0] - scores[:, 1]) / 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(9371)
    with np.load(args.dataset, allow_pickle=False) as data:
        features = torch.tensor(data["x"], dtype=torch.long)
        base = torch.tensor(data["base"], dtype=torch.float32)
        target = torch.tensor(data["target"], dtype=torch.float32)
        validation = torch.tensor(data["valid"], dtype=torch.bool)
    training = torch.where(~validation)[0]
    held_out = torch.where(validation)[0]
    model = Evaluator()
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    best = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(30):
        order = training[torch.randperm(len(training))]
        for batch in torch.split(order, 512):
            prediction = base[batch] + model(features[batch])
            loss = (
                (torch.sigmoid(prediction / 240) - torch.sigmoid(target[batch] / 240)) ** 2
            ).mean()
            optimiser.zero_grad()
            torch.autograd.backward(loss)
            optimiser.step()
        with torch.no_grad():
            correction = torch.cat(
                [model(features[batch]) for batch in torch.split(held_out, 1024)]
            )
            prediction = base[held_out] + correction
            loss_value = float(
                (
                    (torch.sigmoid(prediction / 240) - torch.sigmoid(target[held_out] / 240)) ** 2
                ).mean()
            )
            error = float((prediction - target[held_out]).abs().mean())
            if loss_value < best:
                best = loss_value
                np.savez(
                    args.output,
                    embedding=model.embedding.weight.numpy(),
                    output=model.output.weight.numpy().reshape(-1),
                    bias=model.output.bias.numpy(),
                )
        record = {"epoch": epoch + 1, "validation_loss": loss_value, "validation_mae_cp": error}
        history.append(record)
        print(json.dumps(record), flush=True)
    metadata = {
        "seed": 9371,
        "training_positions": len(training),
        "validation_positions": len(held_out),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
