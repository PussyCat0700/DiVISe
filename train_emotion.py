import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import wandb
from tqdm import tqdm

from dataset.avdatasets import EmotionDataset
from dataset.dataset_loading import load_hifigan_config
from models import EmbeddingClassifier, Generator
from utils import seed_everything


def forward_batch(generator:Generator, batch, device):
    labels = batch['labels'].to(device)
    frame = batch['frames'].to(device)
    units = batch['units'].to(device)
    with torch.no_grad():
        embedding = generator.extract_unit_embedding(units, frame)
    return embedding, labels.squeeze(-1)

def train_model(generator:Generator, model, dataloader, criterion, optimizer, device):
    seed_everything(42)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        embedding, labels = forward_batch(generator, batch, device)
        optimizer.zero_grad()
        outputs = model(embedding)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_description(f'{running_loss/(i+1)}')

    accuracy = 100 * correct / total
    return running_loss/(i+1), accuracy

def test_model(generator, model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader)
        for i, batch in enumerate(pbar):
            embedding, labels = forward_batch(generator, batch, device)
            outputs = model(embedding)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_description(f'{running_loss/(i+1)}')

    accuracy = 100 * correct / total
    return running_loss/(i+1), accuracy

def save_checkpoint(state, is_best, epoch, log_dir):
    """ Save checkpoint and keep the best model based on validation accuracy """
    checkpoint_path = os.path.join(log_dir, f"model-epoch-{epoch}.ckpt")
    
    # Save current checkpoint
    torch.save(state, checkpoint_path)
    
    # If the current model is the best, save it as model-best.ckpt
    if is_best:
        best_path = os.path.join(log_dir, "model-best.ckpt")
        shutil.copyfile(checkpoint_path, best_path)
    
    # Remove previous epoch's checkpoint if it exists
    previous_epoch_path = os.path.join(log_dir, f"model-epoch-{epoch-1}.ckpt")
    if os.path.exists(previous_epoch_path):
        os.remove(previous_epoch_path)

def load_latest_checkpoint(log_dir, model, optimizer=None):
    """ Load the latest checkpoint if available """
    checkpoints = [f for f in os.listdir(log_dir) if f.endswith(".ckpt") and 'best' not in f]
    
    if not checkpoints:
        return 0, 0.0  # No checkpoint found, return model and optimizer unchanged
    
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[2].split('.')[0]))[-1]
    checkpoint_path = os.path.join(log_dir, latest_checkpoint)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint: {latest_checkpoint} at {epoch=}")
    return epoch, checkpoint['best_val_acc']

def load_best_checkpoint(log_dir, model):
    """ Load the best model based on validation accuracy """
    best_checkpoint_path = os.path.join(log_dir, "model-best.ckpt")
    
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded best model checkpoint: {best_checkpoint_path} at {epoch=}")
    else:
        print("No best model checkpoint found!")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train emotion or gender classification model.")
    parser.add_argument('--hifigan_config', default='conf/hifigan/video2speech_revise_original.json')
    parser.add_argument('--task', choices=['emotion', 'gender'], required=True, 
                        help='Task to perform: emotion or gender classification')
    parser.add_argument('--generator_path', required=True, help='Generator path')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for TensorBoard logs and checkpoints')
    
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--use_wandb', action='store_true', help='[Optional] Use Weights & Biases logging')
    parser.add_argument('--use_farl', action='store_true', help='Enable FaRL in model')
    parser.add_argument('--test', action='store_true', help='Skip training and only run test')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generator
    h = load_hifigan_config(args.hifigan_config)
    # Generator loading
    generator = Generator(
                    h,
                    h.num_mels,
                    unit_nums=h.k,
                    use_farl=args.use_farl
                )
    checkpoint = torch.load(os.path.join(args.generator_path, 'model-best.pt'))
    generator.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["generator"]["model"].items()})
    generator = generator.to(device)
    
     # Hyperparameters and setup
    input_size = generator.conv_pre.out_channels
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs

    # Model, criterion, optimizer
    if args.task == 'emotion':
        num_classes = 8  # Number of emotion classes
    elif args.task == 'gender':
        num_classes = 2  # Gender: male or female
    model = EmbeddingClassifier(input_size, num_classes).to(device)
    
    # Dataset and DataLoader (customize based on actual dataset)
    train_dataset = EmotionDataset(task=args.task, split='train', km_pad_class=h.k)
    valid_dataset = EmotionDataset(task=args.task, split='valid', km_pad_class=h.k)
    test_dataset = EmotionDataset(task=args.task, split='test', km_pad_class=h.k)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    criterion = nn.CrossEntropyLoss()

    # TensorBoard setup
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # Optionally initialize WandB
    if args.use_wandb:
        wandb.init(project="ravdess_classification", config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "task": args.task
        })
        wandb.watch(model)

    # Load checkpoint if provided
    start_epoch = 0

    # Test only mode
    if not args.test:
        # Load latest checkpoint if available
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        start_epoch, best_val_acc = load_latest_checkpoint(args.log_dir, model, optimizer)
        patience_counter = 0
        max_patience = 15  # Early stopping after 15 epochs of no improvement
        # Training loop with validation
        for epoch in range(start_epoch + 1, args.epochs + 1):
            train_loss, train_accuracy = train_model(generator, model, train_loader, criterion, optimizer, device)
            valid_loss, valid_accuracy = test_model(generator, model, valid_loader, criterion, device)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

            # Log to TensorBoard
            writer.add_scalar(f'{args.task}/Train_Loss', train_loss, epoch)
            writer.add_scalar(f'{args.task}/Train_Accuracy', train_accuracy, epoch)
            writer.add_scalar(f'{args.task}/Valid_Loss', valid_loss, epoch)
            writer.add_scalar(f'{args.task}/Valid_Accuracy', valid_accuracy, epoch)

            # Optionally log to WandB
            if args.use_wandb:
                wandb.log({
                    'Train Loss': train_loss,
                    'Train Accuracy': train_accuracy,
                    'Valid Loss': valid_loss,
                    'Valid Accuracy': valid_accuracy
                })

            # Save checkpoint if validation accuracy is the best
            is_best = valid_accuracy > best_val_acc
            if is_best:
                best_val_acc = valid_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, is_best, epoch, args.log_dir)
            if patience_counter >= max_patience:
                print(f'early stopping triggered at {max_patience=}')
                break

    # Load best checkpoint
    model = load_best_checkpoint(args.log_dir, model)
    # Test evaluation
    test_loss, test_accuracy = test_model(generator, model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Log test results to TensorBoard and WandB
    writer.add_scalar(f'{args.task}/Test_Loss', test_loss, 0)
    writer.add_scalar(f'{args.task}/Test_Accuracy', test_accuracy, 0)

    if args.use_wandb:
        wandb.log({'Test Loss': test_loss, 'Test Accuracy': test_accuracy})

    writer.close()

if __name__ == "__main__":
    main()
