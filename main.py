
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication

from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch
from torchvision import datasets, transforms
from network.mnist import *
import os.path
import logging


# --------------------- LOGGER ----------------------------------
class QTextEditLogger(logging.Handler):
    def __init__(self, plainTextWidget):
        super().__init__()
        self.widget = plainTextWidget
        self.widget.setReadOnly(True)

    def emit(self, record):
        print(record)
        msg = self.format(record)
        self.widget.appendPlainText(msg)


# --------------------- GUI ------------------------------------
cls, wnd = uic.loadUiType('GUI.ui')


class GUI(wnd, cls):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # --- logger -------------------------------------------
        handler = QTextEditLogger(self.train_log)
        handler.setLevel(logging.INFO)
        self.logger = logging.getLogger()
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # --- device configuration -----------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- training network parameters ----------------------
        self.epochs = 1
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.gamma = 0.7
        self.step_size = 1
        self.accuracy = 0.0

        # --- dataset preparation -------------------------------
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        self.dataset_test = datasets.MNIST('./data', train=False, transform=transform)
        train_kwargs = {'batch_size': self.batch_size_train}
        test_kwargs = {'batch_size': self.batch_size_test}
        if torch.cuda.is_available():
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, **test_kwargs)
        _, (self.example_data, _) = next(enumerate(self.test_loader))

        # --- load saved model -----------------------------------
        if os.path.isfile('./network/results/mnist_cnn.pt'):
            self.model = MnistNet().to(self.device)
            dict = torch.load('./network/results/mnist_cnn.pt')
            self.model.load_state_dict(dict['state_dict'])
            self.train_losses = dict['train_losses']
            self.train_counter = dict['train_counter']
            self.test_losses = dict['test_losses']
            self.test_counter = dict['test_counter']
            self.plt_train_log.draw_loss(self.train_counter, self.train_losses, dict['optimizer_name'])
            self.accuracy = dict['accuracy']
            self.current_optimizer = dict['optimizer_name']
        else:
            self.model = None
            self.train_losses = []
            self.train_counter = []
            self.test_losses = []
            self.test_counter = [i*len(self.train_loader.dataset) for i in range(self.epochs+1)]
        self.cb_optim.setCurrentText('SGD')

    def on_pb_train_released(self):
        # Push button triggering training
        self.train_log.setPlainText('Wait...')
        QApplication.processEvents()
        self.trainer()
        self.acc_log.setPlainText(f'Current model accuracy = {self.accuracy:0.2f} %')

    def on_pb_acc_released(self):
        # Push button triggering evaluation
        self.acc_log.setPlainText('Wait...')
        QApplication.processEvents()
        if self.model is None:
            self.acc_log.setPlainText("There is no trained model.\nClick 'Train the network' first!.")
        else:
            self.accuracy = test(self.model, self.device, self.test_loader, self.test_losses)
            self.acc_log.setPlainText(f'Current model accuracy = {self.accuracy:0.2f} %')

            if self.current_optimizer == 'Adadelta':
                self.acc_log_Adadelta.setPlainText(f'Adadelta accuracy = {self.accuracy:0.2f} %')
            elif self.current_optimizer == 'Adagrad':
                self.acc_log_Adagrad.setPlainText(f'Adagrad accuracy = {self.accuracy:0.2f} %')  
            elif self.current_optimizer == 'Adam':
                self.acc_log_Adam.setPlainText(f'Adam accuracy = {self.accuracy:0.2f} %')
            elif self.current_optimizer == 'RMSprop':
                self.acc_log_RMSprop.setPlainText(f'RMSprop accuracy = {self.accuracy:0.2f} %')
            elif self.current_optimizer == 'SGD':
                self.acc_log_SGD.setPlainText(f'SGD accuracy = {self.accuracy:0.2f} %')

    def on_pb_exam_released(self):
        # Push button to show example inference
        if self.model is None:
            self.acc_log.setPlainText("There is no trained model.\nClick 'Train the network' first!.")
        else:
            with torch.no_grad():
                output = self.model(self.example_data)
                self.plt_example.show_example(self.example_data, output)

    def on_pb_show_last_released(self):
        # Push button triggering training
        if self.train_counter:
            self.plt_train_log.draw_loss(self.train_counter, self.train_losses, 'Last')
        else:
            self.train_log.setPlainText("There is no trained model.\nClick 'Train the network' first!.")
    
    def on_pb_plot_clear_released(self):
        # Push button to clear plot
        self.plt_train_log.clear_plot()

    def on_sb_epochs_valueChanged(self, value):
        # Spin box to change epochs number
        self.epochs = value

    def on_sb_batchsize_valueChanged(self, value):
        # Spin box to change batchsize
        self.batch_size_train = int(value)
        train_kwargs = {'batch_size': self.batch_size_train}
        if torch.cuda.is_available():
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, **train_kwargs)

    def on_sb_lr_valueChanged(self, value):
        # Spin box to change learning rate
        self.learning_rate = value

    def on_sb_gamma_valueChanged(self, value):
        # Spin box to change scheduler gamma
        self.gamma = value

    def on_sb_step_valueChanged(self, value):
        # Spin box to change scheduler step size
        self.step_size = value

    def trainer(self):
        # Function to train, evaluate and save model
        self.model = MnistNet().to(self.device)

        # Set optimizer
        self.current_optimizer = self.cb_optim.currentText()
        if self.current_optimizer == 'Adadelta':
            optimizer = optim.Adadelta(self.model.parameters(), lr=self.learning_rate)
        if self.current_optimizer == 'Adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)   
        if self.current_optimizer == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.current_optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif self.current_optimizer == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        # Clear data
        self.test_losses = []
        self.train_losses = []
        self.train_counter = []
        self.test_counter = [i*len(self.train_loader.dataset) for i in range(self.epochs+1)]

        # Start training for 'epoch'es
        scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        self.accuracy = test(self.model, self.device, self.test_loader, self.test_losses)
        for epoch in range(1, self.epochs + 1):
            train(self.model, self.device, self.train_loader, optimizer, epoch, self.train_losses,
                  self.train_counter, self.logger, flush_log=QApplication.processEvents)
            self.accuracy = test(self.model, self.device, self.test_loader, self.test_losses)
            scheduler.step()

        # Save model
        if self.cb_savemodel.checkState():
            save_dict = {
                'state_dict': self.model.state_dict(),
                'train_counter': self.train_counter,
                'train_losses': self.train_losses,
                'test_counter': self.test_counter,
                'test_losses': self.test_losses,
                'optimizer': optimizer.state_dict(),
                'optimizer_name': self.current_optimizer,
                'accuracy': self.accuracy
            }
            torch.save(save_dict, "./network/results/mnist_cnn.pt")
        
        # Plot train loss vs number of interations
        self.plt_train_log.draw_loss(self.train_counter, self.train_losses, self.current_optimizer)


if __name__ == '__main__':
    app = QApplication([])
    window = GUI()
    window.show()
    app.exec()
