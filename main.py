import torch
from torch.utils.tensorboard import SummaryWriter
from train import per_epoch_activity
import utils
from module import LeNet
from datetime import datetime

batch_size = 4
epochs = 20
# augmentation
aug = utils.image_augmentation()
device = utils.get_device()
# dataset
train_dataset, validation_dataset = utils.get_FashionMNist(aug)
# dataloader
train_dataloader, validation_dataloader = utils.create_dataloader(train_dataset, validation_dataset, batch_size)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Model
model = LeNet(1, 10)
model.to(device)
# loss function
loss_fn = torch.nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# summary writer
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')

if __name__ == '__main__':
    per_epoch_activity(train_dataloader, validation_dataloader, optimizer, model, loss_fn, writer, device,
                       epochs)
    torch.save(model, r'saved_model\model_2epoch.pth')
