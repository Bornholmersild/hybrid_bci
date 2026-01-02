import torch


class FusionNet_train_eval():
    def train_one_epoch(self, model, train_loader, criterion, optimizer, device):
        '''
        Docstring for train_one_epoch

        :param model: Provide the model to be trained
        :param train_loader: Provide the dataloader for training
        :param criterion: Loss function
        :param optimizer: Optimizer for training
        :param device: Device to run the training on (CPU or GPU)

        :return avg_loss : float
        '''
        running_loss = 0.0

        for eeg, emg, lab in train_loader:
            # Every data instance is an input + label pair
            eeg, emg, labels = eeg.to(device), emg.to(device), lab.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            logits, _ = model(eeg = eeg, emg = emg)

            # Compute the loss and its gradients
            loss = criterion(logits, labels)
            loss.backward()                         # Backward pass

            # Adjust learning weights
            optimizer.step()

            # Metrics
            running_loss += loss.item()
                
        avg_loss = running_loss / len(train_loader) # loss per batch

        return avg_loss

    def evaluate_one_epoch(self, model, test_loader, criterion, device):
        '''
        Docstring for evaluate_one_epoch

        :param model: Provide the model to be evaluated
        :param test_loader: Provide the dataloader for testing
        :param criterion: Loss function
        :param device: Device to run the evaluation on (CPU or GPU)

        :return avg_vloss : float, vacc : float, H : list [batch, h_state], Y : list [batch]
        '''
        running_vloss = 0.0
        vcorrect = 0
        vtotal = 0

        L_list = []
        H_list = []
        Y_list = []

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for eeg, emg, lab in test_loader:
                veeg, vemg, vlabels = eeg.to(device), emg.to(device), lab.to(device)

                # Forward pass: compute predicted outputs by passing inputs to the model
                vlogits, h_final = model(eeg = veeg, emg = vemg)

                L_list.append(vlogits.cpu())
                H_list.append(h_final.cpu())
                Y_list.append(vlabels.cpu())

                # Calculate the loss
                vloss = criterion(vlogits, vlabels)

                # Update running validation loss
                running_vloss += vloss.item()
                
                _, vpredicted = torch.max(vlogits, 1)
                vtotal += vlabels.size(0)
                vcorrect += (vpredicted == vlabels).sum().item()

        avg_vloss = running_vloss / len(test_loader) # loss per batch
        vacc = 100 * vcorrect / vtotal
        return avg_vloss, vacc, [L_list, H_list, Y_list]
    
class SingleNet_train_eval():
    def train_one_epoch(self, model, train_loader, criterion, optimizer, device):
        '''
        Docstring for train_one_epoch

        :param model: Provide the model to be trained
        :param train_loader: Provide the dataloader for training
        :param criterion: Loss function
        :param optimizer: Optimizer for training
        :param device: Device to run the training on (CPU or GPU)

        :return avg_loss : float
        '''
        running_loss = 0.0

        for inp, lab in train_loader:
            # Every data instance is an input + label pair
            inputs, labels = inp.to(device), lab.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            logits, _, _ = model(inputs)

            # Compute the loss and its gradients
            loss = criterion(logits, labels)
            loss.backward()                         # Backward pass

            # Adjust learning weights
            optimizer.step()

            # Metrics
            running_loss += loss.item()
                
        avg_loss = running_loss / len(train_loader) # loss per batch

        return avg_loss

    def evaluate_one_epoch(self, model, test_loader, criterion, device):
        '''
        Docstring for evaluate_one_epoch

        :param model: Provide the model to be evaluated
        :param test_loader: Provide the dataloader for testing
        :param criterion: Loss function
        :param device: Device to run the evaluation on (CPU or GPU)

        :return avg_vloss : float, vacc : float, H : list [batch, h_state], Y : list [batch]
        '''
        running_vloss = 0.0
        vcorrect = 0
        vtotal = 0

        H = []
        Y = []

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for inp, lab in test_loader:
                vinputs, vlabels = inp.to(device), lab.to(device)

                # Forward pass: compute predicted outputs by passing inputs to the model
                vlogits, _, vh_final = model(vinputs)

                H.append(vh_final.cpu())
                Y.append(vlabels.cpu())

                # Calculate the loss
                vloss = criterion(vlogits, vlabels)

                # Update running validation loss
                running_vloss += vloss.item()
                
                _, vpredicted = torch.max(vlogits, 1)
                vtotal += vlabels.size(0)
                vcorrect += (vpredicted == vlabels).sum().item()

        avg_vloss = running_vloss / len(test_loader) # loss per batch
        vacc = 100 * vcorrect / vtotal
        return avg_vloss, vacc, H, Y