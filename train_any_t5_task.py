from torch import cuda
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TFT5ForConditionalGeneration
import wandb


def get_device():
    return 'cuda' if cuda.is_available() else 'cpu'


class T5Dataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: T5Tokenizer, source_text_len: int, target_text_len: int,
                 source_attribute: str, target_attribute: str):
        self.__tokenizer = tokenizer

        self.__data = data

        self.__source_text_len = source_text_len

        self.__target_text_len = target_text_len

        self.__source_attribute = source_attribute

        self.__target_attribute = target_attribute

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, index):
        source_text = str(self.__data.iloc[index][self.__source_attribute])

        target_text = str(self.__data.iloc[index][self.__target_attribute])

        source = self.__tokenizer.batch_encode_plus([source_text],
                                                    max_length=self.__source_text_len,
                                                    pad_to_max_length=True,
                                                    truncation=True,
                                                    return_tensors='pt')

        target = self.__tokenizer.batch_encode_plus([target_text],
                                                    truncation=True,
                                                    max_length=self.__target_text_len,
                                                    pad_to_max_length=True,
                                                    return_tensors='pt')

        source_ids = source['input_ids'].squeeze()

        source_mask = source['attention_mask'].squeeze()

        target_ids = target['input_ids'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class Trainer:

    def train(self, epoch: int, tokenizer: T5Tokenizer, data_loader, model, device, optimizer):
        model.train()

        for _, data in enumerate(data_loader, 0):

            y = data['target_ids'].to(device, dtype=torch.long)

            y_ids = y[:, :-1].contiguous()

            lm_labels = y[:, 1:].clone().detach()

            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

            ids = data['source_ids'].to(device, dtype=torch.long)

            mask = data['source_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, lm_labels=lm_labels)

            loss = outputs[0]

            if _ % 10 == 0:
                wandb.log({"Training Loss": loss.item()})

            if _ % 500 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    def validate(self, tokenizer: T5Tokenizer, model, device, loader):
        model.eval()

        predictions = []

        actuals = []

        with torch.no_grad():
            for _, data in enumerate(loader, 0):
                y = data['target_ids'].to(device, dtype=torch.long)

                ids = data['source_ids'].to(device, dtype=torch.long)

                mask = data['source_mask'].to(device, dtype=torch.long)
                
                # TODO make sure you modify the max_length and/or the number of beams
                generated_ids = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=32,
                    num_beams=1,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True
                )

                preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                         generated_ids]

                target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

                if _ % 100 == 0:
                    print(f'Completed {_}')

                predictions.extend(preds)

                actuals.extend(target)

        return predictions, actuals

    def start(self, path_to_training_data: str, delimiter: str = "\t"):
        wandb.init(project="sentence_doctor")

        config = wandb.config

        # TODO Set the batch size during training
        config.TRAIN_BATCH_SIZE = 8

        # TODO Set the batch size at validation time
        config.VALID_BATCH_SIZE = 8

        # TODO Set your training epochs
        config.TRAIN_EPOCHS = 3

        # TODO Set the validation epochs
        config.VAL_EPOCHS = 1

        # TODO Set the
        config.LEARNING_RATE = 0.00005

        # TODO Set your seed
        config.SEED = 42

        # TODO What is the ma length of your input ? (Note that the input is consist of source text and context if training sentence doctor).
        #  Refer to the documentation under Usage if you don't know what is meant by context.
        config.SOURCE_MAX_LEN = 64

        # TODO What is the max length of your output ?
        config.TARGET_MAX_LEN = 32

        torch.manual_seed(config.SEED)

        np.random.seed(config.SEED)

        torch.backends.cudnn.deterministic = True

        # TODO Change the T5 tokenizer. Leave it untouched if you don't have any other one you prefer
        tokenizer = T5Tokenizer.from_pretrained("flexudy/t5-base-multi-sentence-doctor")

        df = pd.read_csv(path_to_training_data, encoding='utf-8', delimiter=delimiter)

        print(df.head())

        # TODO Change the training size. Use a lower training size if you have less data
        train_size = 0.985

        train_dataset = df.sample(frac=train_size, random_state=config.SEED)

        val_dataset = df.drop(train_dataset.index).reset_index(drop=True)

        train_dataset = train_dataset.reset_index(drop=True)

        print("FULL Dataset: {}".format(df.shape))

        print("TRAIN Dataset: {}".format(train_dataset.shape))

        print("TEST Dataset: {}".format(val_dataset.shape))

        # TODO Change the name of the attributes specifying the input text and the output text.
        #  i used source for input and target for output as column names in my pandas dataframe
        training_set = T5Dataset(train_dataset, tokenizer, config.SOURCE_MAX_LEN, config.TARGET_MAX_LEN, "source",
                                 "target")

        val_set = T5Dataset(val_dataset, tokenizer, config.SOURCE_MAX_LEN, config.TARGET_MAX_LEN, "source", "target")

        train_params = {
            'batch_size': config.TRAIN_BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0
        }

        val_params = {
            'batch_size': config.VALID_BATCH_SIZE,
            'shuffle': False,
            'num_workers': 0
        }

        training_loader = DataLoader(training_set, **train_params)

        val_loader = DataLoader(val_set, **val_params)

        # TODO Load the model you want for pretraining. You don't have to use this one. Any T5 LM should work fine
        model = T5ForConditionalGeneration.from_pretrained("flexudy/t5-base-multi-sentence-doctor")

        device = get_device()

        model = model.to(device)

        # TODO Change the optimiser and use whatever works for you. I like AMSGrad
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE, amsgrad=True)

        wandb.watch(model, log="all")

        print('Initiating Fine-Tuning for the model on our dataset')

        for epoch in range(config.TRAIN_EPOCHS):
            self.train(epoch, tokenizer, training_loader, model, device, optimizer)

            for _ in range(config.VAL_EPOCHS):
                predictions, actuals = self.validate(tokenizer, model, device, val_loader)

                final_df = pd.DataFrame({'Repaired': predictions, 'Original': actuals})

                # TODO Change the location of the predictions during evaluation
                final_df.to_csv('./predictions.csv')

                print('Output Files generated for review')

        # TODO Give a cool name to your awesome model
        tokenizer.save_pretrained("t5-base-multi-your-sentence-doctor")

        model.save_pretrained("t5-base-multi-your-sentence-doctor")


if __name__ == '__main__':

    trainer = Trainer()

    # TODO Where is your data ? Enter the path
    trainer.start("data/sentence_doctor_dataset_300.csv")
