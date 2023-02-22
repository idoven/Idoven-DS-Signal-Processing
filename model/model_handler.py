import torch
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from model.model_architecture import ECGNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelHandler():
    def __init__(self, input_channels, signal_length, num_classes, config):
        self.model = ECGNet(input_channels, signal_length, num_classes, config).to(device)
        self.config = config
        mlflow.set_experiment(experiment_name='PTB_XL')
        mlflow.start_run(run_name=config['run_name'])
        mlflow.log_params(config)

    def train(self, train_generator: torch.utils.data.DataLoader):
        logging_interval = 10
        loss_fct = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        for epoch in range(self.config['epochs']):
            # Training
            for x, y in train_generator:
                optimizer.zero_grad()
                x, y = x.to(device).float(), y.to(device).float()
                output = self.model(x)

                loss = loss_fct(output, y)

                loss.backward()
                optimizer.step()
            if epoch % logging_interval == 0:
                print('Epoch ' + str(epoch) + ' Loss: ' + str(loss.cpu().detach().numpy()))
                results = self.validate(train_generator, mode='train')
                mlflow.log_metrics(results, step=epoch)
                print(results)

    def validate(self, test_generator: torch.utils.data.DataLoader, mode: str= 'test'):
        outputs = np.array([])
        gt = []
        with torch.no_grad():
            self.model.eval()
            for x, y in test_generator:
                x, y = x.to(device).float(), y.to(device).float()
                predicted_outputs = self.model(x)
                predicted_outputs = predicted_outputs.cpu().detach().numpy().astype(np.int8).copy()
                y = y.cpu().detach().numpy().astype(np.int8).copy()
                if outputs.shape[0] == 0:
                    outputs = predicted_outputs
                    gt = y
                else:
                    outputs = np.concatenate([outputs, predicted_outputs], axis=0)
                    gt = np.concatenate([gt, y], axis=0)

        results_dict = {
            mode + '_NORM_acc': accuracy_score(gt[:,0], outputs[:,0]),
            mode + '_NORM_recall': recall_score(gt, outputs, labels=[0], average='micro', zero_division=1),
            mode + '_NORM_precision': precision_score(gt, outputs, labels=[0], average='micro', zero_division=1),
            mode + '_MI_acc': accuracy_score(gt[:,1], outputs[:,1]),
            mode + '_STTC_acc': accuracy_score(gt[:,2], outputs[:,2]),
            mode + '_CD_acc': accuracy_score(gt[:,3], outputs[:,3]),
            mode + '_HYP_acc': accuracy_score(gt[:,4], outputs[:,4]),
        }
        if mode == 'test':
            mlflow.log_metrics(results_dict)

        return results_dict

    def end_run(self):
        mlflow.end_run()
