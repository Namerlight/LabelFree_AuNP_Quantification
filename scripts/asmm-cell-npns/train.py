import sklearn.metrics as metrics
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
import numpy as np
from torchsummary import summary
import os
import model_archs
import warnings
from datetime import datetime
import itertools

warnings.filterwarnings("ignore", module="matplotlib\..*")


def train(selected_model, dataset: str, hyperparameters: dict, train_path: str, val_path: str, output_path: str):
    cells_dataset_train = datasets.CellsNPNDataset(dataset_dir=os.path.join("..", "..", train_path),
                                                   transform=datasets.image_transforms(dataset))
    cells_dataset_val = datasets.CellsNPNDataset(dataset_dir=os.path.join("..", "..", val_path),
                                                 transform=datasets.image_transforms(dataset))

    # model = model_archs.FCRNA()
    # model = model_archs.UNet(channels=3, filters=64, kernel_size=3)
    # model = model_archs.TransUNet(img_dim=256, in_channels=3, out_channels=128, head_num=4, mlp_dim=512, block_num=8, patch_dim=16, class_num=1)
    # model = model_archs.TransMCCNN(img_dim=256, in_channels=3, out_channels=64, head_num=4, mlp_dim=512, block_num=8, patch_dim=8, class_num=1)

    model = selected_model

    model_name = model._get_name()
    print("Model: ", model_name)
    batch_size = hyperparameters.get("batch_size")
    epochs = hyperparameters.get("epochs")
    lr = hyperparameters.get("learning_rate")
    momentum = hyperparameters.get("momentum")
    weight_decay = hyperparameters.get("weight_decay")
    criterion = hyperparameters.get("loss_func")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    LRscheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    save_path = os.path.join("..", "..", f"{output_path}", "models",
                             f"{dataset}_{model_name}_b2-16-64_batch_{batch_size}_epochs_{epochs}_lr_{lr}_{datetime.now():%H-%M-%S}.pth")
    plot_save_path = os.path.join("..", "..", f"{output_path}", "plots",
                                  f"{dataset}_{model_name}_b2-16-64_batch_{batch_size}_epochs_{epochs}_lr_{lr}_{datetime.now():%H-%M-%S}")

    train_loader = torch.utils.data.DataLoader(cells_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(cells_dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f"Computing on {device}")

    model.to(device)
    # model_summary = summary(model, (3, 256, 256))
    # print(model_summary)

    total_step = len(train_loader)

    # List of Train and Validation Losses and Errors PER EPOCH
    train_loss, val_loss = [], []
    train_error, val_error = [], []

    valid_loss_min = np.Inf

    for epoch in tqdm(range(epochs)):

        model.train()

        # Setting loss of current epoch to 0.
        running_loss = 0.0

        # List of True and Predicted counts and losses PER BATCH
        true_values_train, predicted_values_train, losses_train = [], [], []
        true_values_val, predicted_values_val, losses_val = [], [], []

        for batch_n, (data, target) in enumerate(train_loader):

            input_data, input_target = data, target
            data, target = data.to(device), target.to(device)
            data, target = data.type(torch.cuda.FloatTensor), target.type(torch.cuda.FloatTensor)

            # Training, so we zero out the gradient.
            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, target)

            # Training, so we update losses and step the optimizer,
            loss.backward()
            optimizer.step()

            # Print Batch information every 10 batches)
            if batch_n % 20 == 0:

                # Display graphs of Input, Target and Output every 5 epochs.
                if epoch % 10 == 0:
                    fig, ax = plt.subplots(3)
                    fig.suptitle(f"Input, Target and Output Grid, Epoch {epoch}")
                    ax[0].imshow(
                        np.transpose(torchvision.utils.make_grid(input_data).detach().cpu().numpy(), (1, 2, 0)),
                        interpolation='nearest')
                    ax[1].imshow(
                        np.transpose(torchvision.utils.make_grid(input_target).detach().cpu().numpy(), (1, 2, 0)),
                        interpolation='nearest')
                    ax[2].imshow(np.transpose(torchvision.utils.make_grid(outputs).detach().cpu().numpy(), (1, 2, 0)),
                                 interpolation='nearest')
                    plt.tight_layout()
                    # plt.show()
                    correct_objects = torch.sum(input_target).item() / (255 * batch_size)
                    n_objects = torch.sum(outputs).item() / (255 * batch_size)
                    print("Actual Count =", correct_objects, "Predicted Count =", n_objects)

            # Update the current epoch's running loss with each batch.
            running_loss += loss.item()

            # Append the batch loss to losses_train
            losses_train.append(loss.item())

            # Iterate through the Targets and Outputs in the batch.
            for true, predicted in zip(target, outputs):
                # Get true count by summing up the true/targets images RGB values for every pixel, divided by 100.
                # Do the same for predicted counts.
                true_counts = torch.sum(true).item() / (255 * batch_size)
                predicted_counts = torch.sum(predicted).item() / (255 * batch_size)

                # Append the list of true values and predicted values for this batch.
                true_values_train.append(true_counts)
                predicted_values_train.append(predicted_counts)

        # Append the average loss over this epoch to the train loss for the epoch
        train_loss.append((running_loss / total_step))
        train_mae = metrics.mean_absolute_error(true_values_train, predicted_values_train)
        train_error.append(train_mae)

        # if epoch < 10:
        #     print(predicted_counts)

        with torch.no_grad():

            model.eval()
            running_loss = 0.0

            for batch_n, (data, target) in enumerate(val_loader):

                data, target = data.to(device), target.to(device)
                data, target = data.type(torch.cuda.FloatTensor), target.type(torch.cuda.FloatTensor)

                outputs = model(data)
                loss = criterion(outputs, target)

                running_loss += loss.item()
                losses_val.append(loss.item())

                for true, predicted in zip(target, outputs):
                    true_counts = torch.sum(true).item() / (255 * batch_size)
                    predicted_counts = torch.sum(predicted).item() / (255 * batch_size)
                    true_values_val.append(true_counts)
                    predicted_values_val.append(predicted_counts)

            val_loss.append((running_loss / total_step))
            val_mae = metrics.mean_absolute_error(true_values_val, predicted_values_val)
            val_error.append(val_mae)

            network_learned = val_loss[-1] < valid_loss_min

            epoch_log = f" Epoch: {epoch} - Train Error: {train_mae} | Train Loss: {np.mean(train_loss)} | Val Error: {val_mae} | Val Loss: {np.mean(val_loss)}"
            tqdm.write(epoch_log)

            if network_learned:
                valid_loss_min = val_loss[-1]
                torch.save(model.state_dict(), save_path)
                # tqdm.write('Improvement-Detected, saving-model')

        LRscheduler.step()

    plot_loss_accuracy(epochs=epochs, true=true_values_train, pred=predicted_values_train, loss=train_loss,
                       error=train_error, save=plot_save_path, mode="Training")
    plot_loss_accuracy(epochs=epochs, true=true_values_val, pred=predicted_values_val, loss=val_loss, error=val_error,
                       save=plot_save_path, mode="Validation")

    print("Training Errors List:", train_error)
    print("Val Errors List:", val_error)


def show_grid(img):
    image = img
    plt.imshow((np.transpose(image, (1, 2, 0))).astype('uint8'), interpolation='nearest')


def plot_loss_accuracy(epochs, true, pred, loss, error, save, mode):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    fig.suptitle(mode, fontsize=14)

    true_line = [[0, max(true)]] * 2  # y = x

    ax1.set_title("Values")
    ax1.set_xlabel('True value')
    ax1.set_ylabel('Predicted value')
    ax1.plot(*true_line, 'r-')
    ax1.scatter(true, pred)

    ax2.set_title("Loss")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.plot(range(epochs), loss)
    plt.tight_layout()

    ax3.set_title("Error")
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE')
    ax3.plot(range(epochs), error)

    plt.tight_layout()
    plt.savefig(f"{save}_{mode}.png")
    plt.show()


if __name__ == '__main__':

    paramlist_batch_size = [8]  # best, final
    paramlist_epochs = [80]  # best, final
    paramlist_learning_rate = [0.5e-2]  # best, final

    paramlist_models = [
        model_archs.MCNNog()
    ]

    params_to_vary = list(
        itertools.product(paramlist_batch_size, paramlist_epochs, paramlist_learning_rate, paramlist_models))

    train_path = os.path.join("data", "asmm", "cell_imgs_npns", "normal")
    val_path = os.path.join("data", "asmm", "cell_imgs_npns", "normal_val")
    output_path = "output_ablation"

    for idx, (bs, ep, lrate, mdl) in enumerate(params_to_vary):
        print("Step", idx, "-", bs, ep, lrate, mdl)

        hyperparameters = {
            "batch_size": bs,
            "epochs": ep,
            "learning_rate": lrate,
            "loss_func": torch.nn.MSELoss(),
            "momentum": 0.9,
            "weight_decay": 1e-5,
        }

        train(selected_model=mdl, dataset="asmm-cell-npns", hyperparameters=hyperparameters, train_path=train_path,
              val_path=val_path, output_path=output_path)

    # Ablation for hyperparameters:
    # Learning rates above 0.01 - model fails to learn
    # Learning rate at 0.005 or 0.001 - best LR
    # 0.1 chance for flips or 5 to 85 degree rotation
    # Low-Low-High (3 3 15)
