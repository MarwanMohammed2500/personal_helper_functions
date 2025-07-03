import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

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
		epochs_count: list, the number of epochs listed (step of 10)
		train_loss_tracker: list, the training loss corresponding to each epoch in epochs_count
		test_loss_tracker: list, the testing loss corresponding to each epoch in epochs_count
	"""
	epochs_count = []
	train_loss_tracker = []
	test_loss_tracker = []
	
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
				epochs_count.append(epoch)
				train_loss_tracker.append(loss.detach().numpy())
				test_loss_tracker.append(test_loss.numpy())
			if verbose:
				print(f"Epoch #{epoch+1:03}, Training Loss = {loss:.4f}, test Loss = {test_loss:.4f}")
	
	# Final evaluation
	model.eval()
	with torch.inference_mode():
		test_preds = model(X_test)
		test_loss = loss_fn(test_preds, y_test)
		epochs_count.append(epochs-1)
		train_loss_tracker.append(loss.detach().numpy())
		test_loss_tracker.append(test_loss.numpy())
		print(f"Final Training Loss = {loss:.4f} | Final test Loss = {test_loss:.4f}")
	return model, epochs_count, train_loss_tracker, test_loss_tracker
	
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
		epochs_count: list, the number of epochs listed (step of 10)
		train_loss_tracker: list, the training loss corresponding to each epoch in epochs_count
		test_loss_tracker: list, the testing loss corresponding to each epoch in epochs_count
	"""
	epochs_count = []
	train_loss_tracker = []
	test_loss_tracker = []
	train_accuracy_tracker = []
								 
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
				epochs_count.append(epoch)
				train_loss_tracker.append(loss.detach().numpy())
				test_loss_tracker.append(test_loss.numpy())
				accuracy = accuracy_score(test_preds, y_test)
				train_accuracy_tracker.append(accuracy)
			if verbose:
				print(f"Epoch #{epoch+1:03}, Training Loss = {loss:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {accuracy:.2%}")
	
	# Final evaluation
	model.eval()
	with torch.inference_mode():
		test_logits = model(X_test)
		test_preds = (torch.sigmoid(test_logits) > thresh).int()
		test_loss = loss_fn(test_logits, y_test)
		epochs_count.append(epochs-1)
		train_loss_tracker.append(loss.detach().numpy())
		test_loss_tracker.append(test_loss.numpy())
		accuracy = accuracy_score(test_preds, y_test)
		train_accuracy_tracker.append(accuracy)
		print(f"Final Training Loss = {loss:.4f} | Final test Loss = {test_loss:.4f} | Final Test Accuracy = {accuracy:.2%}")
	return model, epochs_count, train_loss_tracker, test_loss_tracker
	
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
		epochs_count: list, the number of epochs listed (step of 10)
		train_loss_tracker: list, the training loss corresponding to each epoch in epochs_count
		test_loss_tracker: list, the testing loss corresponding to each epoch in epochs_count
	"""
	epochs_count = []
	train_loss_tracker = []
	test_loss_tracker = []
	
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
	
		# 3. Get the prediction probabilites and calcuate the loss function
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
				epochs_count.append(epoch)
				train_loss_tracker.append(loss.detach().numpy())
				test_loss_tracker.append(test_loss.numpy())
				accuracy = accuracy_score(tset_preds, y_test)
				train_accuracy_tracker.append(accuracy)
				if verbose:
					print(f"Epoch #{epoch+1:03}, Training Loss = {loss:.4f}, test Loss = {test_loss:.4f}, Test Accuracy = {accuracy:.2%}")
	
	# Final evaluation
	model.eval()
	with torch.inference_mode():
		test_logits = model(X_test)
		tset_preds = torch.softmax(train_logits, dim=1).argmax(dim=1)
		test_loss = loss_fn(test_logits, y_test.type(torch.long).squeeze())
		epochs_count.append(epochs-1)
		train_loss_tracker.append(loss.detach().numpy())
		test_loss_tracker.append(test_loss.numpy())
		accuracy = accuracy_score(tset_preds, y_test)
		train_accuracy_tracker.append(accuracy)
		print(f"Final Training Loss = {loss:.4f} | Final test Loss = {test_loss:.4f} | Final Test Accuracy = {accuracy: .2%}")
	return model, epochs_count, train_loss_tracker, test_loss_tracker
