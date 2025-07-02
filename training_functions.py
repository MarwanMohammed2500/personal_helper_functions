import torch


def reg_and_bi_train_test_loop(epochs:int, model:torch.nn.Module, X_train:torch.Tensor, X_test:torch.Tensor, y_train:torch.Tensor, y_test:torch.Tensor, loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, verbose:bool=False, device:str="cpu"):
    """
    **Works for both regression and binary classification problems, NOT Multi-class classification problems**
    This train-test loop used for training models (made specifically for PyTorch models) follows the following steps:
        Forward Pass --> Compute the loss --> optimizer.zero_grad --> Backpropagation --> Optimizer step
    And every 10 epochs, it would evaluate on X_test
    input:
        epochs: int, the number of epochs to train the model
        model: torch.nn.Module, the model to train
        X_train, X_test, y_train, y_test: torch.Tensor, the test and training instances (X for predictors, y for targets)
        loss_fn: torch.nn, the loss function to use when training the model
        optimizer: torch.optim, the optimizer to use when training the model
        verbose: bool, default=False, whether to print out the loss function of the model after each test inference
        device: str, default="cpu", what device to use on the tensors
    returns:
        model: torch.nn.Module, the trained model
        epochs_count: list, the number of epochs listed (step of 10)
        train_loss_tracker: list, the training loss corresponding to each epoch in epochs_count
        test_loss_tracker: list, the testing loss corresponding to each epoch in epochs_count
    """
    epochs_count = []
    train_loss_tracker = []
    test_loss_tracker = []
    for epoch in range(epochs):
        model.train() # Set the model to training mode
        train_preds = model(X_train.to(device)) # Inference on the training set
        loss = loss_fn(train_preds, y_train.to(device)) # compute the loss function

        optimizer.zero_grad() # Zero grad, so that the gradients don't stack
        loss.backward() # Backpropagation
        optimizer.step() # Optimizer step

        model.eval() # Set model to evaluation mode
        if epoch % 10 == 0:
            with torch.inference_mode():
                validation_preds = model(X_test.to(device))
                validation_loss = loss_fn(validation_preds, y_test.to(device))
                epochs_count.append(epoch)
                train_loss_tracker.append(loss.detach().numpy())
                test_loss_tracker.append(validation_loss.numpy())
            if verbose:
                print(f"Epoch #{epoch+1:03}, Training Loss = {loss:.4f}, Validation Loss = {validation_loss:.4f}")

    # Final evaluation
    model.eval()
    with torch.inference_mode():
        validation_preds = model(X_test.to(device))
        validation_loss = loss_fn(validation_preds, y_test.to(device))
        epochs_count.append(epochs)
        train_loss_tracker.append(loss.detach().numpy())
        test_loss_tracker.append(validation_loss.numpy())
        print(f"Final Training Loss = {loss:.4f} | Final Validation Loss = {validation_loss:.4f}")
    return model, epochs_count, train_loss_tracker, test_loss_tracker

def multi_class_train_test_loop(epochs:int, model:torch.nn.Module, X_train:torch.Tensor, X_test:torch.Tensor, y_train:torch.Tensor, y_test:torch.Tensor, loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, verbose:bool=False, device:str="cpu"):
    """
    **Works ONLY on Multi-class classification problems**
    This train-test loop used for training models (made specifically for PyTorch models) follows the following steps:
        Forward Pass --> Compute the loss --> optimizer.zero_grad --> Backpropagation --> Optimizer step
    And every 10 epochs, it would evaluate on X_test
    input:
        epochs: int, the number of epochs to train the model
        model: torch.nn.Module, the model to train
        X_train, X_test, y_train, y_test: torch.Tensor, the test and training instances (X for predictors, y for targets)
        loss_fn: torch.nn, the loss function to use when training the model
        optimizer: torch.optim, the optimizer to use when training the model
        verbose: bool, default=False, whether to print out the loss function of the model after each test inference
        device: str, default="cpu", what device to use on the tensors
    returns:
        model: torch.nn.Module, the trained model
        epochs_count: list, the number of epochs listed (step of 10)
        train_loss_tracker: list, the training loss corresponding to each epoch in epochs_count
        test_loss_tracker: list, the testing loss corresponding to each epoch in epochs_count
    """
    epochs_count = []
    train_loss_tracker = []
    test_loss_tracker = []
    for epoch in range(epochs):
        model.train() # Set the model to training mode
        train_logits = model(X_train.to(device))
        train_preds = torch.softmax(train_logits, dim=1).argmax(dim=1) # Inference on the training set
        loss = loss_fn(train_logits, y_train.type(torch.long).to(device)) # compute the loss function

        optimizer.zero_grad() # Zero grad, so that the gradients don't stack
        loss.backward() # Backpropagation
        optimizer.step() # Optimizer step

        model.eval() # Set model to evaluation mode
        if epoch % 10 == 0:
            with torch.inference_mode():
                validation_preds = model(X_test.to(device))
                validation_loss = loss_fn(validation_preds, y_test.type(torch.long).to(device).squeeze())
                epochs_count.append(epoch)
                train_loss_tracker.append(loss.detach().numpy())
                test_loss_tracker.append(validation_loss.numpy())
                if verbose:
                    print(f"Epoch #{epoch+1:03}, Training Loss = {loss:.4f}, Validation Loss = {validation_loss:.4f}")

    # Final evaluation
    model.eval()
    with torch.inference_mode():
        validation_preds = model(X_test.to(device))
        validation_loss = loss_fn(validation_preds, y_test.type(torch.long).to(device).squeeze())
        epochs_count.append(epoch)
        train_loss_tracker.append(loss.detach().numpy())
        test_loss_tracker.append(validation_loss.numpy())
        print(f"Final Training Loss = {loss:.4f} | Final Validation Loss = {validation_loss:.4f}")
    return model, epochs_count, train_loss_tracker, test_loss_tracker
