import torch
import torch.nn.functional as F
import copy
def knowledge_distillation_train(teacher_model, 
                        student_model, 
                        n_epochs,
                        trainloader,
                        criterion,
                        optimizer,
                        optimizer_params,
                        teacher_percentage = 0.5 ,
                        temperature= 2,):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    optimizer = optimizer(student_model.parameters(),**optimizer_params)
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    student_model.train()  
    training_losses =[]
    for epoch in range(n_epochs):
        running_loss = 0.0

        for inputs, targets in trainloader:
            inputs = inputs.to(device)
            targets=  targets.to(device)
            with torch.no_grad():
                teacher_targets = teacher_model(inputs).to(device)
                teacher_targets = F.softmax(teacher_targets / temperature, dim=1)
                teacher_targets=  teacher_targets.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = student_model(inputs)  # Forward pass
            teacher_loss = criterion(outputs, teacher_targets)  # Compute loss
            absolute_loss = criterion(outputs, targets) 
            total_loss = teacher_loss * teacher_percentage  + (1-teacher_percentage) *absolute_loss
            total_loss.backward()  
            optimizer.step() 
            running_loss += total_loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
      
        print(f'Training Loss: {epoch_loss:.4f}')
        training_losses.append(epoch_loss)
        
    return copy.deepcopy(student_model)
        
