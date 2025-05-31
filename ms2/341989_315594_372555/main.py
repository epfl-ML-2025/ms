import argparse
import time
import numpy as np
import torch
from torchinfo import summary

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
from src.methods.dummy_methods import DummyClassifier
from sklearn.model_selection import KFold

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, y_test = load_data()
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    xtrain = normalize_fn(xtrain, means=np.mean(xtest, axis=0), stds=np.std(xtrain, axis=0))
    xtest = normalize_fn(xtest, means=np.mean(xtest, axis=0), stds=np.std(xtest, axis=0))
    # Make a validation set
    if not args.test:
    ### WRITE YOUR CODE HERE
        validation_split = 0.1  # Example: 10% of the data used for validation
        validation_size = int(xtrain.shape[0] * validation_split)
        xval, yval = xtrain[:validation_size], ytrain[:validation_size]
        xtrain, ytrain = xtrain[validation_size:], ytrain[validation_size:]
    ### WRITE YOUR CODE HERE to do any other data processing
    n_classes = get_n_classes(ytrain)
    
    def cross_validate(model_class, x_data, y_data, k_folds, trainer_kwargs):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        accs, f1s = [], []

        for i, (train_idx, val_idx) in enumerate(kf.split(x_data)):
            print(f"\nFold {i+1}/{k_folds}")
            x_train, y_train = x_data[train_idx], y_data[train_idx]
            x_val, y_val = x_data[val_idx], y_data[val_idx]

            model = model_class()

            if i == 0:
                summary(model)
            trainer = Trainer(model, **trainer_kwargs)
            trainer.fit(x_train, y_train)
            preds = trainer.predict(x_val)

            acc = accuracy_fn(preds, y_val)
            f1 = macrof1_fn(preds, y_val)
            print(f"Fold {i+1} — Accuracy: {acc:.2f}%, F1: {f1:.4f}")
            accs.append(acc)
            f1s.append(f1)

        print(f"\nMean over {k_folds} folds — Accuracy: {np.mean(accs):.2f}%, Macro-F1: {np.mean(f1s):.4f}")

    if args.cv > 1:
        model_classes = {
            "mlp": lambda: MLP(input_size=xtrain.shape[1], n_classes=n_classes),
            "cnn": lambda: CNN(input_channels=3, n_classes=n_classes)
        }
        print(f"Used processing unit : {args.device}")
        model_class = model_classes[args.nn_type]
        if args.nn_type == "cnn":
            xtrain = xtrain.reshape(-1, 3, 28, 28)
        cross_validate(model_class, xtrain, ytrain, args.cv, {
            "lr": args.lr,
            "epochs": args.max_iters,
            "batch_size": args.nn_batch_size,
            "device": args.device
        })
        return
    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    print(f"Used processing unit : {args.device}")
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = MLP(input_size=xtrain.shape[1], n_classes=n_classes)  # Instantiate MLP
    elif args.nn_type == "cnn":
        model = CNN(input_channels=3, n_classes=n_classes)
        xtrain = xtrain.reshape(-1, 3, 28, 28)
        xtest  = xtest .reshape(-1, 3, 28, 28)
    else:
        model = DummyClassifier(0)

    summary(model)

    # Trainer object
    
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=args.device)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    start_time = time.perf_counter()  # Start timing
    preds_train = method_obj.fit(xtrain, ytrain)
    training_time = time.perf_counter() - start_time  # End timing
    print(f"\nTraining completed in {training_time:.2f} seconds.")

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, y_test)
    macrof1 = macrof1_fn(preds, y_test)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.

if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!
    # MS2 arguments
    parser.add_argument('--cv', type=int, default=0, help="number of folds for cross-validation (0 = no cross-validation)")
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.

    args = parser.parse_args()
    args.device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() and torch.version.cuda else "cpu")
    main(args)
