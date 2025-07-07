import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

def regression_train_test_loop(epochs:int, model:torch.nn.Module,
                    X_train: torch.Tensor, X_test: torch.Tensor, y_train:torch.Tensor, y_test:torch.Tensor,
                    loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    verbose:bool=False, device:str="cpu"):
    """
    **Train-Test loop implementation for regression problems**
    This train-test loop used for training models (made specifically for PyTorch models) follows the following steps:
    Forward Pass --> Compute the loss --> optimizer.zero_grad --> Backpropagation --> Optimizer step
    And every 10 epochs, it would evaluate on X_test
    Arguments:
    epochs: int, the number of epochs to train the model
    model: torch.nn.Module, the model to train
    X_train, X_test, y_train, y_test: torch.Tensor, the test and training instances (X for predictors, y for targets)
    loss_fn: torch.nn, the loss function to use when training the model
    optimizer: torch.optim, the optimizer to use when training the model
    verbose: bool, default=False, whether to print out the loss function of the model after each test inference
    device: str, default="cpu", what device to use on the tensors
    returns:
    model: torch.nn.Module, the trained model
    tracker: pd.DataFrame, tracked performance (train and test loss) across epochs
    """
    # Send tensors and the model to the proper device
    X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)
    model.to(device)
    tracker = {}
                                
    for epoch in tqdm(range(epochs)):
        # 1. Set the model's mode to "train"
        model.train()

        # 2. Forward Pass 
        train_preds = model(X_train)

        # 3. Calculate the loss function
        loss = loss_fn(train_preds, y_train)

        # 4. Optimizer zero grad
        optimizer.zero_grad() # So that the gradients don't stack

        # 5. Backpropagation
        loss.backward()

        # 6. Optimizer step
        optimizer.step()

        # Model evaluation
        model.eval() # Set model to evaluation mode
        if epoch % 10 == 0:
            with torch.inference_mode():
                test_preds = model(X_test)
                test_loss = loss_fn(test_preds, y_test)
                tracker[epoch] = {"train_loss": loss.item(), "test_loss": test_loss}
            if verbose:
                print(f"Epoch #{epoch+1:03}, Training Loss = {loss:.4f}, test Loss = {test_loss:.4f}")

    # Final evaluation
    model.eval()
    with torch.inference_mode():
        test_preds = model(X_test)
        test_loss = loss_fn(test_preds, y_test)
        tracker[epochs-1] = {"train_loss": loss.item(), "test_loss": test_loss}

        print(f"Final Training Loss = {loss:.4f} | Final test Loss = {test_loss:.4f}")
    return model, pd.DataFrame(tracker).T.astype(float)

def binary_classification_train_test_loop(epochs:int, model:torch.nn.Module,
                    X_train: torch.Tensor, X_test: torch.Tensor, y_train:torch.Tensor, y_test:torch.Tensor,
                    loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    verbose:bool=False, device:str="cpu", thresh: int=0.5):
    """
    **Train-Test loop implementation for binary classification problems**
    This train-test loop used for training models (made specifically for PyTorch models) follows the following steps:
    Forward Pass --> Compute the loss --> optimizer.zero_grad --> Backpropagation --> Optimizer step
    And every 10 epochs, it would evaluate on X_test
    Arguments:
    epochs: int, the number of epochs to train the model
    model: torch.nn.Module, the model to train
    X_train, X_test, y_train, y_test: torch.Tensor, the test and training instances (X for predictors, y for targets)
    loss_fn: torch.nn, the loss function to use when training the model
    optimizer: torch.optim, the optimizer to use when training the model
    verbose: bool, default=False, whether to print out the loss function of the model after each test inference
    device: str, default="cpu", what device to use on the tensors
    thresh: int, default=0.5, the threshold to use when rounding the prediction probabilities
    returns:
    model: torch.nn.Module, the trained model
    tracker: pd.DataFrame, tracked performance (train and test loss) across epochs
    """
    tracker = {}	   
                                
    # Send tensors and the model to the proper device
    X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)      
    model.to(device)
                                
    for epoch in tqdm(range(epochs)):
        # 1. Set the model's mode to "train"
        model.train()

        # 2. Forward Pass
        train_logits = model(X_train) 

        # 3. Get the prediction probabilities and calculate the loss function
        train_preds_probas = torch.sigmoid(train_logits) # Inference on the training set
        loss = loss_fn(train_logits, y_train) # compute the loss function

        # 4. Optimizer zero grad
        optimizer.zero_grad() # So that the gradients don't stack

        # 5. Backpropagation
        loss.backward()

        # 6. Optimizer step
        optimizer.step()

        # Model evaluation
        model.eval() # Set model to evaluation mode
        if epoch % 10 == 0:
            with torch.inference_mode():
                test_logits = model(X_test)
                test_preds = (torch.sigmoid(test_logits) > thresh).int()
                test_loss = loss_fn(test_logits, y_test)
                tracker[epoch] = {"train_loss": loss.item(), "test_loss": test_loss}
            if verbose:
                print(f"Epoch #{epoch+1:03}, Training Loss = {loss:.4f}, Test Loss = {test_loss:.4f}")

    # Final evaluation
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_preds = (torch.sigmoid(test_logits) > thresh).int()
        test_loss = loss_fn(test_logits, y_test)
        tracker[epochs-1] = {"train_loss": loss.item(), "test_loss": test_loss}

        print(f"Final Training Loss = {loss:.4f} | Final test Loss = {test_loss:.4f}")
    return model, pd.DataFrame(tracker).T.astype(float)

def multi_class_train_test_loop(epochs:int, model:torch.nn.Module,
                    X_train: torch.Tensor, X_test: torch.Tensor, y_train:torch.Tensor, y_test:torch.Tensor,
                    loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    verbose:bool=False, device:str="cpu"):
    """
    **Train-Test loop implementation for multi-class classification problems**
    This train-test loop used for training models (made specifically for PyTorch models) follows the following steps:
    Forward Pass --> Compute the loss --> optimizer.zero_grad --> Backpropagation --> Optimizer step
    And every 10 epochs, it would evaluate on X_test
    Arguments:
    epochs: int, the number of epochs to train the model
    model: torch.nn.Module, the model to train
    X_train, X_test, y_train, y_test: torch.Tensor, the test and training instances (X for predictors, y for targets)
    loss_fn: torch.nn, the loss function to use when training the model
    optimizer: torch.optim, the optimizer to use when training the model
    verbose: bool, default=False, whether to print out the loss function of the model after each test inference
    device: str, default="cpu", what device to use on the tensors
    returns:
    model: torch.nn.Module, the trained model
    tracker: pd.DataFrame, tracked performance (train and test loss) across epochs
    """
    tracker = {}

    # Send tensors and the model to the proper device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)         
    model.to(device)
    for epoch in tqdm(range(epochs)):
        # 1. Set the model's mode to "train"
        model.train()

        # 2. Forward Pass
        train_logits = model(X_train)

        # 3. Get the prediction probabilities and calculate the loss function
        train_preds_probas = torch.softmax(train_logits, dim=1).argmax(dim=1) # Inference on the training set
        loss = loss_fn(train_logits, y_train.type(torch.long)) # compute the loss function

        # 4. Optimizer zero grad
        optimizer.zero_grad() # So that the gradients don't stack

        # 5. Backpropagation
        loss.backward()

        # 6. Optimizer step
        optimizer.step() # Optimizer step

        # Model Evaluation
        model.eval() # Set model to evaluation mode
        if epoch % 10 == 0:
            with torch.inference_mode():
                test_logits = model(X_test)
                tset_preds = torch.softmax(train_logits, dim=1).argmax(dim=1)
                test_loss = loss_fn(test_logits, y_test.type(torch.long).squeeze())
                tracker[epoch] = {"train_loss": loss.item(), "test_loss": test_loss}
                if verbose:
                    print(f"Epoch #{epoch+1}, Training Loss = {loss:.4f}, test Loss = {test_loss:.4f}")

    # Final evaluation
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        tset_preds = torch.softmax(train_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_test.type(torch.long).squeeze())
        tracker[epochs-1] = {"train_loss": loss.item(), "test_loss": test_loss}

        print(f"Final Training Loss = {loss:.4f} | Final Test Loss = {test_loss:.4f}")
    return model, pd.DataFrame(tracker).T.astype(float)

def minibatch_binary_class_train_test_loop(model:torch.nn, loss_fn:torch.nn, optimizer:torch.optim.Optimizer, epochs:int,
                                    train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader,
                                    device:str="cpu"):
    """
    **Train-Test loop implementation for batch binary classification**
    This train-test loop used for training models (made specifically for PyTorch models) follows the following steps:
    Forward Pass --> Compute the loss --> optimizer.zero_grad --> Backpropagation --> Optimizer step
    And every 10 epochs, it would evaluate on test_dataloader
    Arguments:
    epochs: int, the number of epochs to train the model
    model: torch.nn.Module, the model to train
    train_dataloader, test_dataloader: torch.utils.data.DataLoader, the test and training DataLoaders
    loss_fn: torch.nn, the loss function to use when training the model
    optimizer: torch.optim, the optimizer to use when training the model
    verbose: bool, default=False, whether to print out the loss function of the model after each test inference
    device: str, default="cpu", what device to use on the tensors
    returns:
    model: torch.nn.Module, the trained model
    tracker: pd.DataFrame, tracked performance (train and test loss) across epochs
    """
    tracker = {}
    for epoch in tqdm(range(epochs)):
        print(f"\n----------------------------Epoch #{epoch}\n")
        train_loss, test_loss = 0, 0

        model.to(device) # Cast the model to the proper device
        model.train()
        for batch, (X_train_batch, y_train_batch) in enumerate(train_dataloader):
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device) # Cast the batch to device
            
            # 1. Forward Pass
            y_pred = model(X_train_batch).squeeze()
            loss = loss_fn(y_pred, y_train_batch)
            train_loss += loss.item()

            # 2. Optimizer Zero Grad
            optimizer.zero_grad()

            # 3. Backpropagation
            loss.backward()

            # 4. Optimzier Step
            optimizer.step()
        
        if epoch % 10 == 0:
            model.eval() # Set model to evaluation mode 
            with torch.inference_mode():
                for X_train_batch, y_train_batch in test_dataloader:
                    X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device) # Cast the batch to device
                    test_pred = model(X_train_batch).squeeze() # prediction
                    test_loss = loss_fn(test_pred, y_train_batch) # loss function
                    test_loss += train_loss

                tracker[epoch] = {"train_loss":train_loss/len(train_dataloader),
                                "test_loss":test_loss/len(test_dataloader)} # Track the train and test loss per epoch
            print(f"Epoch #{epoch}, Average Train Loss = {train_loss/len(train_dataloader):.3f} | Average Validation Loss = {test_loss/len(test_dataloader):.3f}")

    # Final Test Evaluation
    model.eval() # Set model to evaluation mode 
    with torch.inference_mode():
        test_features, test_labels = next(iter(test_dataloader))
        test_features, test_labels = test_features.to(device), test_labels.to(device)
      
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device) # Cast the data to device
        test_pred = model(test_features).squeeze() # prediction
        test_loss = loss_fn(test_pred, test_labels) # loss function
    print(f"\nValidation Loss = {test_features}")

    return model, pd.DataFrame(tracker).T.astype(float)
