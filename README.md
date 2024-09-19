| Parameter           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `teacher_model`     | The pre-trained teacher model used for knowledge distillation.              |
| `student_model`     | The student model that will learn from the teacher model.                   |
| `n_epochs`          | The number of epochs to train the student model.                            |
| `trainloader`       | The DataLoader providing the training data.                                 |
| `criterion`         | The loss function used to compute the loss.                                 |
| `optimizer`         | The optimizer class used to update the model parameters (e.g., `torch.optim.Adam`). |
| `optimizer_params`  | A dictionary of hyperparameters for the optimizer (e.g., `{'lr': 0.001}`).  |
| `teacher_percentage`| The percentage of teacher model's output to be used in the loss calculation. Default is 0.5. |
| `temperature`       | The temperature parameter for softening the logits. Default is 2.           |


Returns the trained student model