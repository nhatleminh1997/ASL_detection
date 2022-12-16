import torch.nn as nn
from asl_dataset import *
from RNN_classifer import *
from torch.utils.data import DataLoader
import numpy as np
def main():
    data_path = r'D:\ML_final\Data'
    use_gpu = True 
    input_size = 28
    sequence_length = 28
    n_classes = 24
    hidden_size = 128
    n_epochs = 80
    learning_rate = 0.01
    num_layers = 2
    batch_size = 32
    n_workers = 4
    # Initialize model
    model = RNN_Classifier(input_size, hidden_size, n_classes, num_layers, batch_first=True, use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()
    
    # Initialize data loaders
    train_set = ASL(data_path, 'Train')
    test_set = ASL(data_path, 'Test')
    n_trains = len(train_set)
    n_tests = len(test_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers= n_workers)

    # Initialize optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Train and evaluate
    train_loss_list = []
    eval_loss_list = []
    eval_accuracy_list = []
    for epoch in range(n_epochs):
        train_loss = 0
        print("epoch: ", epoch)
        ### Training 
        print("training")
        for i, (images, labels) in enumerate(train_loader):
            
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            images = images.reshape(-1, sequence_length, input_size).float()
            outputs = model(images)
            labels = nn.functional.one_hot(labels, n_classes).float()
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().numpy().item()
        train_loss_list.append(train_loss/n_trains)

        print ('Training Loss: {:.4f}' 
                       .format(train_loss_list[-1]))
        ### Testing
        eval_loss = 0
        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                labels_onehot = nn.functional.one_hot(labels, n_classes).float()
                images = images.reshape(-1, sequence_length, input_size).float()
                outputs = model(images)
                loss = criterion(outputs, labels_onehot)
                prediction = torch.argmax(outputs, dim=1)
                correct = correct + (prediction == labels).sum().item()
                eval_loss += loss.detach().cpu().numpy().item()
        eval_loss_list.append(eval_loss/n_tests)
        eval_accuracy_list.append(correct/n_tests)
        print ('Testing Loss: {:.4f}' 
                       .format(eval_loss_list[-1]))
        print ('Prediction accuracy: {:.4f}' 
                       .format(eval_accuracy_list[-1]))
        if np.max(eval_accuracy_list) == eval_accuracy_list[-1]:
            torch.save(model, 'rnn_lr_'+str(learning_rate)+ '_'+str(epoch) + '_' + str(np.round(eval_accuracy_list[-1],4)) + '.pt')
    print(train_loss_list)
    print(eval_loss_list)
    print(eval_accuracy_list)
    


        
    print("-"*20)


if __name__ == "__main__":
    main()