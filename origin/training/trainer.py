import random
import torch

def seed_everything(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)

def make_batches(token_ids: list[int], block_size: int, batch_size: int, device: str, generator: torch.Generator):
    if len(token_ids) <= block_size: raise ValueError("Dataset needs more tokens than block_size.")
    data = torch.tensor(token_ids, dtype=torch.long)
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[start:start + block_size] for start in starts])
    y = torch.stack([data[start + 1:start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate(model, token_ids, block_size, batch_size, batches, device, generator):
    model.eval(); values = []
    for _ in range(batches):
        x, y = make_batches(token_ids, block_size, batch_size, device, generator)
        _, loss = model(x, y); values.append(float(loss.item()))
    return sum(values) / len(values)

def train_steps(model, optimizer, train_ids, val_ids, *, steps, block_size, batch_size, device="cpu", seed=42, eval_batches=2):
    if steps < 1 or batch_size < 1 or eval_batches < 1: raise ValueError("steps, batch_size and eval_batches must be positive.")
    model.to(device); rng = torch.Generator().manual_seed(seed)
    metrics = {"train_loss": [], "validation_loss": []}
    try:
        for _ in range(steps):
            model.train(); x, y = make_batches(train_ids, block_size, batch_size, device, rng)
            optimizer.zero_grad(set_to_none=True); _, loss = model(x, y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            metrics["train_loss"].append(float(loss.detach().item()))
            metrics["validation_loss"].append(evaluate(model, val_ids or train_ids, block_size, batch_size, eval_batches, device, rng))
    except RuntimeError as exc:
        if "memory" in str(exc).lower() or "alloc" in str(exc).lower(): raise MemoryError("Training ran out of memory. Reduce batch_size, n_embd or block_size.") from exc
        raise
    return metrics
