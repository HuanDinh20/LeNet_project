import torch
from datetime import datetime


def train_one_epoch(train_dataloader, optimizer, model, loss_fn, epoch_idx, summary_writer, device):
    """
    in one epoch:
    1. Gets a batch of training data from the DataLoader
    2. Zeros the optimizer’s gradients
    3. Performs an inference - that is, gets predictions from the model for an input batch
    4. Calculates the loss for that set of predictions vs. the labels on the dataset
    5. Calculates the backward gradients over the learning weights
    6. Tells the optimizer to perform one learning step - that is, adjust the model’s learning weights based on the
    observed gradients for this batch, according to the optimization algorithm we chose
    7. It reports on the loss for every 1000 batches.
    8. Finally, it reports the average per-batch loss for the last 1000 batches, for comparison with a validation run
    """
    running_loss = 0.0
    last_loss = 0.0

    # loop to track the batch index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        # 1. get data
        inputs, labels = data
        # 1.1 move input and labels to device
        inputs, labels = inputs.to(device), labels.to(device)
        # 2. zeros the gradients
        optimizer.zero_grad()

        # 3. Performs an inference
        outputs = model(inputs)

        # 4. Compute loss
        loss = loss_fn(outputs, labels)

        # 5. gradients
        loss.backward()

        # 6. adjust the model’s learning weights
        optimizer.step()

        # 7. Gather data and report - every 1000 batches.
        running_loss += loss.item()

        if not (i % 1000):
            last_loss = running_loss / 1000  # loss per batch
            print(f"Batch {i + 1} running_loss {running_loss}")
            print(f"Batch {i + 1} loss {last_loss}")
            tb_x = epoch_idx * len(train_dataloader) + i + 1
            summary_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0
    return last_loss


def per_epoch_activity(train_dataloader, val_dataloader, optimizer, model, loss_fn, summary_writer, device,
                       epochs=5):
    """
    at each epoch:
    1. Perform validation by checking our relative loss on a set of data that was not used for training, and report this
    2. Save a copy of the model
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0
    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print(f"Epoch {epoch_number + 1}")

        model.train(True)
        avg_loss = train_one_epoch(train_dataloader, optimizer, model, loss_fn, epoch_number, summary_writer, device)

        # We don't need gradients for reporting
        model.train(False)
        running_val_loss = 0.0

        for i, val_data in enumerate(val_dataloader):
            val_inputs, val_labels = val_data
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = loss_fn(val_outputs, val_labels)
            running_val_loss += val_loss

        avg_val_loss = running_val_loss / (i + 1)
        print(f"Loss train {avg_loss} valid {avg_val_loss}")

        # Log the running loss averaged per batch
        # for both training and validation

        summary_writer.add_scalars('Training vs. Validation Loss',
                                   {'Training': avg_loss, 'Validation': avg_val_loss},
                                   epoch_number + 1)

        summary_writer.flush()

        # Save the best performance model's state
        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            model_path = fr'saved_model\model_{epoch_number}_{timestamp}'
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
