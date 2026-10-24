"""grounding / head training."""

from __future__ import annotations

import time

import torch

LEARNING_RATES = (0.001, 0.003, 0.01)
EPOCHS = 20


def fit_head(
    output,
    training_rows,
    valid,
    model_factory,
    seed,
    metric,
    *,
    device="cpu",
    learning_rates=LEARNING_RATES,
    epochs=EPOCHS,
    batch_size=128,
    weight_decay=0.0001,
    gradient_clip=1.0,
    budget_seconds=None,
    on_epoch=None,
    collate,
    loss_fn,
    predict,
):
    """Select across rates and epochs, keeping the first result on a metric tie.

    The caller owns input/output setup, model initialization and metric aggregation.
    Only the small cached-head fit belongs here; no upstream inference is performed.
    """
    best, selected, runs, started = -1.0, None, [], time.monotonic()
    for lr in learning_rates:
        torch.manual_seed(seed)
        model = model_factory().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        generator = torch.Generator().manual_seed(seed)
        history = []
        for epoch in range(epochs + 1):
            losses = []
            if epoch:
                model.train()
                order = torch.randperm(len(training_rows), generator=generator).tolist()
                for start in range(0, len(order), batch_size):
                    batch = collate(
                        [training_rows[i] for i in order[start : start + batch_size]],
                        device,
                    )
                    optimizer.zero_grad()
                    loss = loss_fn(model, batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            predictions, _ = predict(model.eval(), valid, device)
            value = metric(predictions, valid)
            history.append(
                dict(
                    epoch=epoch,
                    valid_task_macro_EA=value,
                    train_loss=sum(losses) / len(losses) if losses else None,
                )
            )
            if value > best + 1e-12:
                best = value
                selected = dict(lr=lr, epoch=epoch, valid_task_macro_EA=value)
                torch.save(model.state_dict(), output / "head.pt")
            if (
                budget_seconds is not None
                and time.monotonic() - started > budget_seconds
            ):
                raise RuntimeError(f"Small-head fit exceeded {budget_seconds} seconds")
            if on_epoch is not None:
                on_epoch(lr, epoch, value, selected, time.monotonic() - started)
        runs.append(dict(lr=lr, history=history))
    model.load_state_dict(
        torch.load(output / "head.pt", map_location=device, weights_only=True)
    )
    predictions, records = predict(model.eval(), valid, device)
    assert abs(metric(predictions, valid) - best) < 1e-12
    return model, dict(
        selected=selected,
        runs=runs,
        validation_records=records,
        seconds=time.monotonic() - started,
    )
