import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_epoch(model, iterator, optimizer, criterion):
    model.train()
    running_loss = 0
    losses = []

    for i, batch in enumerate(iterator):
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred.transpose(1, 2), batch.target.T)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        curr_loss = loss.data.cpu().detach().item()
        loss_smoothing = i / (i + 1)
        running_loss = loss_smoothing * running_loss + (1 - loss_smoothing) * curr_loss
    return running_loss, losses


def _test_epoch(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    losses = []

    n_batches = len(iterator)
    with torch.no_grad():
        for batch in iterator:
            pred = model(batch)
            loss = criterion(pred.transpose(1, 2), batch.target.T)
            losses.append(loss.item())
            epoch_loss += loss.data.item()

    return epoch_loss / n_batches, losses


def mle_train(model, train_iterator, valid_iterator, criterion, optimizer, n_epochs=10, early_stopping=0):
    prev_loss = 100500
    es_epochs = 0
    train_losses = []
    valid_losses = []

    for epoch in range(n_epochs):
        train_loss, epoch_tl = _train_epoch(model, train_iterator, optimizer, criterion)
        valid_loss, epoch_vl = _test_epoch(model, valid_iterator, criterion)
        train_losses.extend(epoch_tl)
        valid_losses.extend(epoch_vl)
        print('validation loss %.5f' % valid_loss)

        if early_stopping > 0:
            if valid_loss > prev_loss:
                es_epochs += 1
            else:
                es_epochs = 0

            if es_epochs >= early_stopping:
                print('Early stopping!')
                break
            prev_loss = min(prev_loss, valid_loss)
    return train_losses, valid_losses