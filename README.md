This work was partially support by the ``Trustworthy And Resilient Decentralised Intelligence For Edge Systems (TaRDIS)" Project, funded by EU HORIZON EUROPE program, under grant agreement No 101093006

Input
| Parameter           | Type                | Description                                                                 |
|---------------------|---------------------|-----------------------------------------------------------------------------|
| `teacher_model`     | `torch.nn.Module`   | The pre-trained teacher model used for knowledge distillation.              |
| `student_model`     | `torch.nn.Module`   | The student model that will learn from the teacher model.                   |
| `n_epochs`          | `int`               | The number of epochs to train the student model.                            |
| `trainloader`       | `torch.utils.data.DataLoader` | The DataLoader providing the training data.                                 |
| `criterion`         | `torch.nn`   | The loss function used to compute the loss.                                 |
| `optimizer`         | `torch.optim`       | The optimizer class used to update the model parameters (e.g., `torch.optim.Adam`). |
| `optimizer_params`  | `dict`              | A dictionary of hyperparameters for the optimizer (e.g., `{'lr': 0.001}`).  |
| `teacher_percentage`| `float`             | The percentage of teacher model's output to be used in the loss calculation. Default is 0.5. |
| `temperature`       | `float`             | The temperature parameter for softening the logits. Default is 2.           |

Returns 
| Parameter           | Type            |  Description                                                                 |
|---------------------|-----------------|------------------------------------------------------------|
| `student_model`     |  `torch.nn.Module`|The trained student model                |
| `training_losses`     |   `List`       |    A list with the training losses per epoch          |

