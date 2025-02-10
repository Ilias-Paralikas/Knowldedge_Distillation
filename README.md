# Simple Implementation Of Knowledge Distillation for Classification Task

see demo.py for sample usage
## Parameters

| Parameter           | Type                           | Description                                                                    |
|--------------------|--------------------------------|--------------------------------------------------------------------------------|
| `teacher_model`    | `torch.nn.Module`             | The pre-trained teacher model used for knowledge distillation                  |
| `student_model`    | `torch.nn.Module`             | The student model that will learn from the teacher model                       |
| `trainloader`      | `torch.utils.data.DataLoader`  | The DataLoader providing the training data                                    |
| `criterion`        | `torch.nn.Module`             | The loss function used to compute the loss                                     |
| `optimizer`        | `torch.optim.Optimizer`        | The optimizer used to update the model parameters (e.g., `torch.optim.Adam`)  |
| `teacher_percentage`| `float`                       | The percentage of teacher model's output in loss calculation (default: 0.5)    |
| `temperature`      | `float`                       | The temperature parameter for softening the logits (default: 2)               |

## Returns

| Parameter    | Type    | Description                                  |
|-------------|---------|----------------------------------------------|
| `epoch_loss`| `int`   | A list with the training of one epoch    |

