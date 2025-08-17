import torch
import torch.nn as nn
import torch.optim as optim


import config
from model import MNIST_CNN
from dataset import get_data_loaders

if __name__ == '__main__':

    train_loader, test_loader = get_data_loaders()

    # moves model to cpu
    model = MNIST_CNN().to(config.DEVICE)
    print("Kiến trúc model:\n", model)

    # loss function and optimize
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


    for epoch in range(config.EPOCHS):
        model.train()
        for images, labels in train_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)        #  input through models
            loss = criterion(outputs, labels) #  measure loss
            optimizer.zero_grad()
            loss.backward()               
            optimizer.step()              

        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {loss.item():.4f}")


    model.eval() #evaluate mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f'Accuracy on testing: {acc:.2f} %')

    # save result
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"✅ Model are saved '{config.MODEL_SAVE_PATH}'")

# guarantee model in eval mode
model.eval()

# Example input to trace model execution
dummy_input = torch.randn(1, 1, 28, 28, device=config.DEVICE)

#export
torch.onnx.export(model,
                  dummy_input,
                  config.ONNX_MODEL_PATH,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})

print(f"Model are exported to ONNX '{config.ONNX_MODEL_PATH}'")