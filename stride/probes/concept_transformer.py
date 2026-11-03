"""concept-transformer: the learning-rate schedule.

The project builds its scheduler in CTCModel.configure_optimizers, with
lightning-bolts' LinearWarmupCosineAnnealingLR. Lightning 2.x needs that
library gone, so agents re-implement the schedule. The tests never call this
method; this probe does, and records the learning rate at every epoch.
"""


def probe():
    import types

    import torch
    from ctc.ctc_model import CTCModel

    out = {}
    # The project's defaults (warmup 50 of 150 epochs), and a short run.
    for warmup, max_epochs in [(50, 150), (3, 12)]:
        param = torch.nn.Parameter(torch.zeros(1))
        stub = types.SimpleNamespace(
            parameters=lambda: [param],
            hparams=types.SimpleNamespace(learning_rate=4e-4, weight_decay=1e-3,
                                          warmup=warmup, max_epochs=max_epochs,
                                          disable_lr_scheduler=False))
        optimizers, schedulers = CTCModel.configure_optimizers(stub)
        optimizer = optimizers[0]
        scheduler = schedulers[0]
        if isinstance(scheduler, dict):
            scheduler = scheduler["scheduler"]
        rates = []
        for _ in range(max_epochs):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        out[f"warmup{warmup}_epochs{max_epochs}"] = {
            "optimizer": type(optimizer).__name__,
            "weight_decay": optimizer.param_groups[0]["weight_decay"],
            "lr_by_epoch": rates,
        }
    return out
