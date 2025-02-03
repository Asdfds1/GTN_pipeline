import os.path
from Dataset_prep.Dataloader import GraphMLDataLoader
import torch
from Trainer_and_Models import Model_trainer, custom_criterion
from Dataset_prep.Parsers import GraphMLParser
import numpy as np
import pandas as pd

def get_true_values(data_loader, device):
    """Собирает истинные значения из DataLoader"""
    true_values = []
    for data in data_loader:
        true_values.extend(data.y.squeeze().cpu().numpy())
    return np.array(true_values)

if __name__ == "__main__":

#TODO Нужно использовать random_seed, пока что думаю где его добавить и как

    data_path = '../CircuitGen_AI/datasets/2_0 (50-69 in, 50-100 out)'
    batch_size = 2
    parser = GraphMLParser()
    encoder = 'One-hot-encoder'
    shuffle = True
    n_workers = 10
    use_scaler = False

    loader = GraphMLDataLoader(dataset_path= data_path,
                               batch_size=batch_size,
                               parser=parser,
                               encoder=encoder,
                               shuffle=shuffle,
                               num_workers=n_workers,
                               use_scaler=use_scaler)
    t_loader, v_loader = loader.get_val_train_dataloaders(val_size=0.2)
    all_y_values = []
    for batch in t_loader:
        all_y_values.extend(batch.y.tolist())
    print(f"All y values: {all_y_values}")

    first_batch = next(iter(t_loader))
    input_dim_x = first_batch.x.shape[1]
    hidden_channels = 32
    num_heads = 4
    learning_rate = 0.01
    device = torch.device('cuda')
    criterion = custom_criterion.MAPE_loss()

    transformer_trainer = Model_trainer.TransformerTrainer(in_channels=input_dim_x,
                                                           hidden_channels=hidden_channels,
                                                           num_heads=num_heads,
                                                           learning_rate=learning_rate,
                                                           device=device,
                                                           criterion=criterion)
    # 6. Обучение моделей

    num_epochs = 10

    # transformer_trainer.fit(train_loader=t_loader,
    #                         valid_loader=v_loader,
    #                         num_epochs=num_epochs)
    path = '../CircuitGen_AI/Run_result/20250202_161357_TransformerGNN_MAPE_loss()'
    transformer_trainer.load_best_model(path)
    predictions = transformer_trainer.predict(v_loader)
    true_values = get_true_values(v_loader, device=device)

    print(true_values.shape)
    print(predictions.shape)

    df = pd.DataFrame({
        'true': true_values,
        'predicted': predictions
    })

    save_path = os.path.join(transformer_trainer.run_dir, 'validation_predictions.csv')
    df.to_csv(save_path, index=False)


