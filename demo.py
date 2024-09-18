import torch
import torch.nn as nn
from models import myVGG,vgg19_bn
from knowledge_distilation import knowledge_distillation_train
from torch.utils.data import Dataset, DataLoader

# Define a dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size=100, num_features=10):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), torch.randint(0,10,())



student = myVGG()
teacher = vgg19_bn()
epochs  =1
dummy_dataset = DummyDataset()
trainloader = DataLoader(dummy_dataset, batch_size=10, shuffle=True)
optimizer_params = {'lr': 0.001}
new_model = knowledge_distillation_train(teacher, 
                                         student,
                                         n_epochs=epochs,
                                         trainloader=trainloader,
                                         criterion=nn.CrossEntropyLoss(),
                                        optimizer= torch.optim.Adam,
                                            optimizer_params=optimizer_params,
                                         teacher_percentage=1,
                                         temperature=1)
