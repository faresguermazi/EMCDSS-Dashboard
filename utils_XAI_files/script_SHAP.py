import pandas as pd
import numpy as np
import h5py
import re
import json
import torch
import pickle
import joblib
import rootutils
from pathlib import Path
from transformers import AutoTokenizer
from pandarallel import pandarallel
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, Optional, Tuple

data_dir = '/workspace/XAI/data/mortality' #change the PATH to the data/mortality folder
# the below code will load the data from the test split!!!
data = h5py.File(f'{data_dir}/splits.hdf5', 'r')

# change the key to "with_discharge" or "with_notes" based on requirement
test_data_notes = data["with_notes"]['test']
test_data_discharge = data["with_discharge"]['test']

# the icustay numbers and their input ids
icus_notes = test_data_notes['icu'][()]
input_ids_notes = test_data_notes['input_ids'][()]
icus_discharge = test_data_discharge['icu'][()]
input_ids_discharge = test_data_discharge['input_ids'][()]

# we create index to icustay dict and icustay to index dict
idxtoicu_notes = {i: c for i, c in enumerate(icus_notes)}
icutoidx_notes = {c: i for i, c in enumerate(icus_notes)}

idxtoicu_discharge = {i: c for i, c in enumerate(icus_discharge)}
icutoidx_discharge = {c: i for i, c in enumerate(icus_discharge)}


from pathlib import Path
from transformers import AutoTokenizer
from pandarallel import pandarallel
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, Optional, Tuple

class MIMICDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MIMIC-III benchmark is a dataset created from the MIMIC-III database, which consist of 17 timeseries features captured over a 48 hour period.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        mimic_dir: str = '/workspace/XAI/mimic-iii-clinical-database-1.4', # This needs to be changed in future ###
        data_dir: str = "/workspace/XAI/in_hospital_mortality/", # This needs to be changed in future####
        task_dir: str='/workspace/XAI/data/mortality/', # This needs to be changed in future#####
        duration: float = 48.0,
        batch_size: int = 30,
        notes: bool = True,
        discrete: bool = False,
        discharge: bool = False,
        time_series: bool = True,
        num_workers: int = 1,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MIMICDataModule`.
        :param mimic_dir: The MIMIC-III directory. Defaults to `/netscratch/ravichandran/mimic-iii-clinical-database-1.4`.
        :param data_dir: The data directory. Defaults to `"data/in-hospital-mortality"`.
        :param task_dir: The task directory. Defaults to `/data/mortality`.
        :param duration: The duration. Defaults to `48.0`.
        :param batch_size: The batch size. Defaults to `30`.
        :param notes: Whether to include notes. Defaults to `False`.
        :param discrete: Whether to include discrete. Defaults to `False`.
        :param discharge: Whether to include discharge. Defaults to `True`.
        :param time_series: Whether to include time series. Defaults to `True`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()
        assert time_series or (notes and not discharge) or (not notes and discharge), "Either notes or discharge or time series should be True"
        assert not(discrete and not notes), "Discrete should only be True if notes is True"

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self attributes
        self.mimic_dir = Path(mimic_dir)
        self.data_dir = Path(data_dir)
        self.duration = duration
        self.task_dir = Path(task_dir)
        self.notes = notes
        self.discrete = discrete
        self.discharge = discharge
        self.time_series = time_series
        self.num_workers = num_workers


        with open(self.task_dir / 'onehotencoder.pkl', 'rb') as f:
            self.one_hotencoder = joblib.load(f)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(self.data_dir / ts_filename, "r") as tsfile:
            _ = tsfile.readline()
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return np.stack(ret)
    def read_ts_files(self):
        labels = pd.read_csv(self.task_dir / 'label_file.csv')[['stay','ICUSTAY_ID','partition']]
        labels['location'] = labels['partition'].map(lambda x: 'test' if x == 'test' else 'train')
        labels['location'] = labels['location'] + '/' + labels['stay']
        labels['ts'] = labels['location'].map(self._read_timeseries)
        labels = labels.drop(columns=['partition','location'])
        return labels

    def xs_feats(self):

        data = self.read_ts_files()
        discretizer = Discretizer(config_path=self.task_dir / 'discretizer_config.json')
        data['X'] = data['ts'].map(discretizer.transform)
        data = data.drop(columns='ts')

        print('Length of time series data = ' + str(len(data)))

        stays = pd.read_csv(self.task_dir / 'all_stays.csv')[['ICUSTAY_ID','GENDER', 'AGE', 'ETHNICITY']]

        print('Length of stays data = ' + str(len(stays)))

        data_merge = pd.merge(data, stays, on='ICUSTAY_ID', how='inner')

        print('Length of inner merge data = ' + str(len(data_merge)))

        age = data_merge['AGE'].to_numpy().reshape(-1, 1)

        one_hot_value = self.one_hotencoder.transform(data_merge[['GENDER', 'ETHNICITY']])

        data_merge['s'] = np.hstack((one_hot_value, age)).tolist()

        data_merge = data_merge.drop(columns=['GENDER', 'AGE', 'ETHNICITY'])

        return data_merge
    def xs_hdf5(self):
        if not Path.is_file(self.task_dir / 'Xs.hdf5'):
            xs_feats = self.xs_feats()
            xs_feats.to_hdf(self.task_dir / 'Xs.hdf5', 'Xs')

    @staticmethod
    def tokenize(id_texts, notes):
        pandarallel.initialize(nb_workers=2)

        def preprocess1(x):
            y = re.sub('\\[(.*?)]', '', x)
            y = re.sub('[0-9]+\.', '', y)
            y = re.sub('dr\.', 'doctor', y)
            y = re.sub('m\.d\.', 'md', y)
            y = re.sub('admission date:', '', y)
            y = re.sub('discharge date:', '', y)
            y = re.sub('--|__|==', '', y)
            return y

        def preprocessing(df_less_n):
            df_less_n['TEXT'] = df_less_n['TEXT'].fillna(' ')
            df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\n', ' ', regex=True)
            df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\r', ' ', regex=True)
            df_less_n['TEXT'] = df_less_n['TEXT'].apply(str.strip)
            df_less_n['TEXT'] = df_less_n['TEXT'].str.lower()

            df_less_n['TEXT'] = df_less_n['TEXT'].apply(lambda x: preprocess1(x))
            return df_less_n

        id_texts = preprocessing(id_texts)
        if notes:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            tokenizer = AutoTokenizer.from_pretrained('bvanaken/CORe-clinical-outcome-biobert-v1')

        print(id_texts.shape)

        id_texts['FEATS'] = id_texts['TEXT'].parallel_apply(
            lambda x: tokenizer.encode_plus(text=x, max_length=512, padding='max_length', truncation=True,
                                            return_token_type_ids=True, return_attention_mask=True))

        id_texts = id_texts.drop(columns='TEXT')

        id_texts['input_ids'] = id_texts['FEATS'].apply(lambda x: x['input_ids'])

        id_texts['token_type_ids'] = id_texts['FEATS'].apply(lambda x: x['token_type_ids'])

        id_texts['attention_mask'] = id_texts['FEATS'].apply(lambda x: x['attention_mask'])

        id_texts = id_texts.drop(columns='FEATS')
        return id_texts

    def note_feats(self):
        labels = pd.read_csv(self.task_dir / 'label_file.csv').rename(columns={'y_true': 'LABEL'})

        notes = pd.read_csv(self.mimic_dir / 'NOTEEVENTS.csv', parse_dates=['CHARTDATE', 'CHARTTIME', 'STORETIME'])

        icus = pd.read_csv(self.mimic_dir / 'ICUSTAYS.csv', parse_dates=['INTIME', 'OUTTIME']).sort_values(
            by=['SUBJECT_ID']).reset_index(drop=True)

        df = pd.merge(icus[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME']], notes, on=['SUBJECT_ID', 'HADM_ID'],
                      how='inner')
        df = df.drop(columns=['SUBJECT_ID', 'HADM_ID'])

        df = pd.merge(labels['ICUSTAY_ID'], df, on='ICUSTAY_ID', how='left')

        df = df[df['ISERROR'].isnull()]

        df = df[df['CHARTTIME'].notnull()]

        df['TIME'] = (df['CHARTTIME'] - df['INTIME']).apply(lambda x: x.total_seconds()) / 3600
        df = df[(df['TIME'] <= self.duration) & (df['TIME'] >= 0.0)]
        if self.discrete:
            df['TIME'] = df['TIME'].map(int)
            id_texts = df.groupby(['ICUSTAY_ID', 'TIME'])['TEXT'].apply(lambda x: '\n'.join(x)).reset_index()
            id_texts = self.tokenize(id_texts, self.notes)

            def func(x, length, dim):
                def stack(name):
                    src = np.zeros((length, dim), dtype=np.int64)
                    idx = x['TIME'].to_numpy()
                    arr = np.stack(x[name].to_numpy())
                    src[idx] = arr
                    return src

                return pd.Series({
                    'TIME': x['TIME'].to_numpy(),
                    'input_ids': stack('input_ids'),
                    'token_type_ids': stack('token_type_ids'),
                    'attention_mask': stack('attention_mask')
                })

            id_texts = id_texts.groupby('ICUSTAY_ID')[['TIME', 'input_ids', 'token_type_ids', 'attention_mask']]. \
                parallel_apply(func, length=int(self.duration), dim=512)
            id_texts = id_texts.drop(columns='TIME').reset_index()
            return id_texts
        else:

            df['VARNAME'] = df[['CATEGORY', 'DESCRIPTION']].apply(lambda x: x.CATEGORY + '/' + x.DESCRIPTION, axis=1)
            df = df.groupby(['ICUSTAY_ID', 'VARNAME'])[['TIME', 'TEXT']].apply(lambda x: x.iloc[x['TIME'].argmax()]) # last clinical notes in the 48 hours duration is only considered

            id_texts = df.groupby('ICUSTAY_ID')['TEXT'].apply(lambda x: '\n'.join(x)).reset_index()
            return self.tokenize(id_texts, self.notes)

    def note_hdf5(self):
        if self.discrete:
            if not Path.is_file(self.task_dir / 'discrete_notes.hdf5'):
                discrete_feats = self.note_feats()
                discrete_feats['ICUSTAY_ID'].to_hdf(self.task_dir / 'discrete_notes.hdf5', 'ICUSTAY_ID')
                discrete_feats['input_ids'].to_hdf(self.task_dir / 'discrete_notes.hdf5', 'input_ids')
                discrete_feats['token_type_ids'].to_hdf(self.task_dir / 'discrete_notes.hdf5', 'token_type_ids')
                discrete_feats['attention_mask'].to_hdf(self.task_dir / 'discrete_notes.hdf5', 'attention_mask')
        else:
            if not Path.is_file(self.task_dir / 'notes.hdf5'):
                note_feats = self.note_feats()
                note_feats.to_hdf(self.task_dir / 'notes.hdf5', 'notes')
    def discharge_feats(self):

        # load dataframes
        mimic_notes = pd.read_csv(self.mimic_dir / 'NOTEEVENTS.csv')
        mimic_admissions = pd.read_csv(self.mimic_dir / "ADMISSIONS.csv")
        mimic_icustay = pd.read_csv(self.mimic_dir / 'ICUSTAYS.csv')

        # filter notes
        mimic_notes = filter_notes(mimic_notes, mimic_admissions, mimic_icustay)

        # filter out written out death indications
        mimic_notes_filter_death = remove_death(mimic_notes)

        return self.tokenize(mimic_notes_filter_death, self.notes)

    def discharge_hdf5(self):
        if not Path.is_file(self.task_dir / 'discharge.hdf5'):
            discharge_feats = self.discharge_feats()
            discharge_feats = discharge_feats.drop(columns=['HADM_ID', 'ROW_ID', 'SUBJECT_ID', 'CHARTDATE', 'CHIEF_COMPLAINT','PRESENT_ILLNESS', 'MEDICAL_HISTORY',
                                                            'MEDICATION_ADM', 'ALLERGIES', 'PHYSICAL_EXAM', 'FAMILY_HISTORY', 'SOCIAL_HISTORY'])
            discharge_feats.to_hdf(self.task_dir / 'discharge.hdf5', 'discharge')

    def save_data(self, file, name, splits, index, df_label):
        hf = h5py.File(file, 'a')
        group = hf.create_group(name)
        split_idx = zip(splits, index)
        for s, idx in split_idx:
            s = group.create_group(s)
            s.create_dataset('X', data=np.stack(df_label.loc[idx]['X'].to_numpy()))
            s.create_dataset('s', data=np.stack(df_label.loc[idx]['s'].to_numpy()))
            s.create_dataset('icu', data=np.stack(df_label.loc[idx]['ICUSTAY_ID'].to_numpy()))
            if self.notes or self.discharge:
                if self.discrete:
                    s.create_dataset('time', data=np.stack(df_label.loc[idx]['TIME'].to_numpy()))
                s.create_dataset('input_ids', data=np.stack(df_label.loc[idx]['input_ids'].to_numpy()))
                s.create_dataset('token_type_ids',
                                 data=np.stack(df_label.loc[idx]['token_type_ids'].to_numpy()))
                s.create_dataset('attention_mask',
                                 data=np.stack(df_label.loc[idx]['attention_mask'].to_numpy()))
            s.create_dataset('label', data=np.stack(df_label.loc[idx]['LABEL'].to_numpy()))

    def split(self, file, k):
        df_label = pd.read_csv(self.task_dir / 'label_file.csv').rename(columns={'y_true': 'LABEL'})

        if self.notes or self.discharge:

            if self.notes and not self.discharge:
                if self.discrete:
                    text_feats = pd.read_hdf(self.task_dir / 'notes.hdf5', 'discrete_notes')
                else:
                    text_feats = pd.read_hdf(self.task_dir / 'notes.hdf5', 'notes')
            elif not self.notes and self.discharge:
                text_feats = pd.read_hdf(self.task_dir / 'discharge.hdf5', 'discharge')


            df_label = pd.merge(df_label, text_feats, on='ICUSTAY_ID', how='left')

            df_label = df_label[df_label['input_ids'].notnull()]

        df_subjects = df_label.reset_index().merge(pd.read_hdf(self.task_dir / 'Xs.hdf5', 'Xs'), on='ICUSTAY_ID',
                                                   how='left').set_index('index')

        train_idx = df_subjects[df_subjects['partition'] == 'train'].index.values
        print("No.of train examples : " + str(len(train_idx)))
        val_idx = df_subjects[df_subjects['partition'] == 'val'].index.values
        print("No.of val examples : " + str(len(val_idx)))
        test_idx = df_subjects[df_subjects['partition'] == 'test'].index.values
        print("No.of test examples : " + str(len(test_idx)))
        partitions = ['train', 'val', 'test']
        indices = [train_idx, val_idx, test_idx]
        self.save_data(file, k, partitions, indices, df_subjects)

    def split_hdf5(self):
        file = self.task_dir / 'splits.hdf5'

        if self.notes and not self.discharge:
            if self.discrete:
                k = 'with_discrete_notes'
            else:
                k = 'with_notes'
        elif self.discharge and not self.notes:
            k = 'with_discharge'
        else:
            k = 'without_notes'

        if not Path.is_file(file):
            self.split(file, k)
        else:
            try:
                with h5py.File(file, 'r') as hf:
                    hf[k]
            except KeyError:
                self.split(file, k)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        if self.notes and not self.discharge:
            self.note_hdf5()
        if not self.notes and self.discharge:
            self.discharge_hdf5()

        self.xs_hdf5()

        self.split_hdf5()

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            hf = h5py.File(self.task_dir / 'splits.hdf5', 'r')
            if self.notes and not self.discharge:
                if self.discrete:
                    group = hf['with_discrete_notes']
                else:
                    group = hf['with_notes']
            elif not self.notes and self.discharge:
                group = hf['with_discharge']
            else:
                group = hf['without_notes']

            self.data_train = EHRDataset(group['train'], self.notes, self.discrete, self.discharge, self.time_series)
            self.data_val = EHRDataset(group['val'], self.notes, self.discrete, self.discharge, self.time_series)
            self.data_test = EHRDataset(group['test'], self.notes, self.discrete, self.discharge, self.time_series)

            hf.close()

    def selected_sample(self, icu_id):

        df_label = pd.read_csv(self.task_dir / 'label_file.csv').rename(columns={'y_true': 'LABEL'})

        if self.notes and not self.discharge:
            if self.discrete:
                text_feats = pd.read_hdf(self.task_dir / 'notes.hdf5', 'discrete_notes')
            else:
                text_feats = pd.read_hdf(self.task_dir / 'notes.hdf5', 'notes')
        elif not self.notes and self.discharge:
            text_feats = pd.read_hdf(self.task_dir / 'discharge.hdf5', 'discharge')

        df_label = pd.merge(df_label, text_feats, on='ICUSTAY_ID', how='inner')

        df_label = df_label[df_label['input_ids'].notnull()]

        df_label = df_label[df_label['ICUSTAY_ID'] == icu_id]

        df_label['file_location'] = df_label['partition'] + '/' + df_label['stay']

        file_name = df_label['file_location'].to_list()[0]

        time_series = self._read_timeseries(file_name)

        text_ids = df_label['input_ids']

        return time_series.T.tolist(), text_ids.to_list()[0]

    def selected_sampletodisk(self, icu_id):

        file_type = 'note' if self.notes else 'discharge'

        file_name = '{}_{}.txt'.format(icu_id, file_type)

        file_path = self.task_dir / "annotation_sample" / file_name

        time_series, input_ids = self.selected_sample(icu_id)

        if self.notes:
            print('special token is skipped')
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            print('special token is skipped')
            tokenizer = AutoTokenizer.from_pretrained('bvanaken/CORe-clinical-outcome-biobert-v1')

        header = ['Capillary_refill_rate',
          'Diastolic_blood_pressure',
          'Fraction_inspired_oxygen',
          'Glascow_coma_scale_eye_opening',
          'Glascow_coma_scale_motor_response',
          'Glascow_coma_scale_total',
          'Glascow_coma_scale_verbal_response',
          'Glucose',
          'Heart_Rate',
          'Height',
          'Mean_blood_pressure',
          'Oxygen_saturation',
          'Respiratory_rate',
          'Systolic_blood_pressure',
          'Temperature',
          'Weight',
          'pH']

        with open(file_path, "w") as file:
            file.write("ICUSTAY-ID: " + str(icu_id) + '\n')
            file.write(tokenizer.decode(input_ids, skip_special_tokens=True))
            file.write("\n\n")
            for i, item in enumerate(time_series[1:]):
                file.write(header[i] + ': ' + ', '.join([str(round(float(x), 2)) if '.' in x and not x.isdigit() and not 'ET/Trach' in x else x for x in item])
            + '\n\n')

        print('written {} to disk'.format(file_name))

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

        # data preprocessing code
class Discretizer:
    def __init__(self, timestep=1, store_masks=True, impute_strategy='zero', start_time='zero', end=48,
                 config_path=None):
        assert config_path is not None

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']


        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self.end = end
        self._impute_strategy = impute_strategy

        self._header = ["Hours"] + self._id_to_channel
        self._encoded_header = self._header_list()

        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def _header_list(self):
        # create new header
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        return new_header

    def transform(self, X):
        eps = 1e-6
        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        max_hours = self.end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            if self._is_categorical_channel[channel]:
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)

        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = self._header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1

                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values

        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins - 1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        return data


def filter_notes(notes_df, admissions_df, icu_stay_df):
    """
    Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
    their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
    """
    # filter out newborns
    adm_grownups = admissions_df[admissions_df.ADMISSION_TYPE != "NEWBORN"]
    notes_df = notes_df[notes_df.HADM_ID.isin(adm_grownups.HADM_ID)]

    # adding icu stay id to the data
    notes_df = pd.merge(icu_stay_df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']], notes_df, on=['SUBJECT_ID', 'HADM_ID'],
                  how='inner')

    # remove notes with no TEXT or HADM_ID
    notes_df = notes_df.dropna(subset=["TEXT", "ICUSTAY_ID", "HADM_ID"])

    # filter discharge summaries
    notes_df = notes_df[notes_df.CATEGORY == "Discharge summary"]

    # remove duplicates and keep the later ones
    notes_df = notes_df.sort_values(by=["CHARTDATE"])
    notes_df = notes_df.drop_duplicates(subset=["TEXT"], keep="last")

    # combine text of same admissions (those are usually addendums)
    combined_adm_texts = notes_df.groupby('HADM_ID')['TEXT'].apply(lambda x: '\n\n'.join(x)).reset_index()
    notes_df = notes_df[notes_df.DESCRIPTION == "Report"]
    notes_df = notes_df[["HADM_ID", "ROW_ID", "SUBJECT_ID", "ICUSTAY_ID", "CHARTDATE"]]
    notes_df = notes_df.drop_duplicates(subset=["HADM_ID"], keep="last")
    notes_df = pd.merge(combined_adm_texts, notes_df, on="HADM_ID", how="inner")

    # strip texts from leading and trailing and white spaces
    notes_df["TEXT"] = notes_df["TEXT"].str.strip()

    # remove entries without admission id, subject id or text
    notes_df = notes_df.dropna(subset=["HADM_ID", "SUBJECT_ID", "TEXT"])

    notes_df = filter_admission_text(notes_df)

    return notes_df

def filter_admission_text(notes_df) -> pd.DataFrame:
    """
    Filter text information by section and only keep sections that are known on admission time.
    """
    admission_sections = {
        "CHIEF_COMPLAINT": "chief complaint:",
        "PRESENT_ILLNESS": "present illness:",
        "MEDICAL_HISTORY": "medical history:",
        "MEDICATION_ADM": "medications on admission:",
        "ALLERGIES": "allergies:",
        "PHYSICAL_EXAM": "physical exam:",
        "FAMILY_HISTORY": "family history:",
        "SOCIAL_HISTORY": "social history:"
    }

    # replace linebreak indicators
    notes_df['TEXT'] = notes_df['TEXT'].str.replace(r"\n", r"\\n", regex=True)

    # extract each section by regex
    for key in admission_sections.keys():
        section = admission_sections[key]
        notes_df[key] = notes_df.TEXT.str.extract(r'(?i){}(.+?)\\n\\n[^(\\|\d|\.)]+?:'
                                                  .format(section))

        notes_df[key] = notes_df[key].str.replace(r'\\n', r' ', regex=True)
        notes_df[key] = notes_df[key].str.strip()
        notes_df[key] = notes_df[key].fillna("")
        # notes_df[notes_df[key].str.startswith("[]")][key] = ""
        notes_df.loc[notes_df[key].str.startswith("[]"), key] = ""

    # filter notes with missing main information
    notes_df = notes_df[(notes_df.CHIEF_COMPLAINT != "") | (notes_df.PRESENT_ILLNESS != "") |
                        (notes_df.MEDICAL_HISTORY != "")]

    # add section headers and combine into TEXT_ADMISSION
    notes_df = notes_df.assign(TEXT="CHIEF COMPLAINT: " + notes_df.CHIEF_COMPLAINT.astype(str)
                                    + '\n\n' +
                                    "PRESENT ILLNESS: " + notes_df.PRESENT_ILLNESS.astype(str)
                                    + '\n\n' +
                                    "MEDICAL HISTORY: " + notes_df.MEDICAL_HISTORY.astype(str)
                                    + '\n\n' +
                                    "MEDICATION ON ADMISSION: " + notes_df.MEDICATION_ADM.astype(str)
                                    + '\n\n' +
                                    "ALLERGIES: " + notes_df.ALLERGIES.astype(str)
                                    + '\n\n' +
                                    "PHYSICAL EXAM: " + notes_df.PHYSICAL_EXAM.astype(str)
                                    + '\n\n' +
                                    "FAMILY HISTORY: " + notes_df.FAMILY_HISTORY.astype(str)
                                    + '\n\n' +
                                    "SOCIAL HISTORY: " + notes_df.SOCIAL_HISTORY.astype(str))

    return notes_df

def remove_death(df: pd.DataFrame):
    """
    Some notes contain mentions of the patient's death such as 'patient deceased'. If these occur in the sections
    PHYSICAL EXAM and MEDICATION ON ADMISSION, we can simply remove the mentions, because the conditions are not
    further elaborated in these sections. However, if the mentions occur in any other section, such as CHIEF COMPLAINT,
    we want to remove the whole sample, because the patient's passing if usually closer described in the text and an
    outcome prediction does not make sense in these cases.
    """

    death_indication_in_special_sections = re.compile(
        r"((?:PHYSICAL EXAM|MEDICATION ON ADMISSION):[^\n\n]*?)((?:patient|pt)?\s+(?:had\s|has\s)?(?:expired|died|passed away|deceased))",
        flags=re.IGNORECASE)

    death_indication_in_all_other_sections = re.compile(
        r"(?:patient|pt)\s+(?:had\s|has\s)?(?:expired|died|passed away|deceased)", flags=re.IGNORECASE)

    # first remove mentions in sections PHYSICAL EXAM and MEDICATION ON ADMISSION
    df['TEXT'] = df['TEXT'].replace(death_indication_in_special_sections, r"\1", regex=True)

    # if mentions can be found in any other section, remove whole sample
    df = df[~df['TEXT'].str.contains(death_indication_in_all_other_sections)]

    # remove other samples with obvious death indications
    df = df[~df['TEXT'].str.contains("he expired", flags=re.IGNORECASE)]  # does also match 'she expired'
    df = df[~df['TEXT'].str.contains("pronounced expired", flags=re.IGNORECASE)]
    df = df[~df['TEXT'].str.contains("time of death", flags=re.IGNORECASE)]

    return df

class EHRDataset(Dataset):
    def __init__(self, split, notes, discrete, discharge, time_series):
        self.notes = notes
        self.discrete = discrete
        self.discharge = discharge
        self.time_series = time_series

        self.X = split['X'][()]
        self.s = split['s'][()]
        self.icu = split['icu'][()]

        self.y = split['label'][()]
        assert len(self.X) == len(self.s) and len(self.X) == len(self.y) and len(self.X) == len(self.icu)
        if self.notes or self.discharge:
            if self.discrete:
                self.time = split['time'][()]
            self.input_ids = split['input_ids'][()]
            self.token_type_ids = split['token_type_ids'][()]
            self.attention_mask = split['attention_mask'][()]
            assert len(self.input_ids) == len(self.y)

    def __getitem__(self, index):
        xi = self.X[index]
        si = self.s[index]
        icus = self.icu[index]
        y = torch.tensor(self.y[index]).float()
        L, _ = xi.shape

        if (not self.notes and not self.discharge) and self.time_series:
            x = torch.from_numpy(xi).float()
            return x, y, icus
        else:
            if self.discrete:
                base = torch.zeros((L, self.input_ids[0].shape[-1]))
                input_ids = torch.scatter(base, 0, self.time[index], self.input_ids[index])
                token_type_ids = torch.scatter(base, 0, self.time[index], self.token_type_ids[index])
                attention_mask = torch.scatter(base, 0, self.time[index], self.attention_mask[index])
            else:
                input_ids = torch.tensor(self.input_ids[index])
                token_type_ids = torch.tensor(self.token_type_ids[index])
                attention_mask = torch.tensor(self.attention_mask[index])

            if self.time_series:
                si = torch.from_numpy(si).float()
                xi = torch.from_numpy(xi).float()
                return input_ids, si, xi, token_type_ids, attention_mask, y, icus
            else:
                return input_ids, token_type_ids, attention_mask, y, icus


    def __len__(self):
        return len(self.y)


from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, BertConfig, BertModel

# The below code defines the different models
class Lstm(nn.Module):
    def __init__(self,
                 input_size: int = 76,
                 hidden_size: int = 64,
                 n_neurons: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.project = nn.Linear(hidden_size * 2, n_neurons)
        self.drop = nn.Dropout(dropout)

    def forward(self, X):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(X)
        output = lstm_out[:, -1, :]
        return F.relu(self.drop(self.project(output)))

class Bert(nn.Module):
    def __init__(self,
                 pretrained_bert: str = '',
                 att_out: bool = False,
                 freeze: tuple = ()):
        super().__init__()
        self.att_out = att_out
        config = BertConfig(vocab_size=30522)

        if self.att_out:
            self.bert = BertModel(config=config, output_hidden_states=True, output_attentions=True)
        else:
            self.bert = BertModel(config=config)

        for name, param in self.bert.named_parameters():
            param.requires_grad = True
            for layer in freeze:
                if layer in name:
                    param.requires_grad = False
                    break

    def forward(self, input_ids, token_type_ids, attention_mask):
        if self.att_out:
            output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            return output.pooler_output, output.hidden_states, output.attentions
        else:
            return self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output

class TBert(nn.Module):
    def __init__(self,
                 pretrained_bert: str = 'bvanaken/CORe-clinical-outcome-biobert-v1',
                 att_out: bool = False,
                 bert_size: int = 768,
                 output_size: int = 1,) -> None:
        super().__init__()
        self.enc = Bert(pretrained_bert=pretrained_bert, att_out=att_out)
        self.pred = nn.Linear(bert_size, output_size)
    def forward(self, input_ids, token_type_ids, attention_mask):

        if self.enc.att_out:
            nt, hidden, att = self.enc(input_ids, token_type_ids, attention_mask)
            return torch.sigmoid(self.pred(nt)).squeeze(dim=-1), hidden, att
        else:
            nt = self.enc(input_ids, token_type_ids, attention_mask)
            return torch.sigmoid(self.pred(nt)).squeeze(dim=-1)


class Gate(nn.Module):
    def __init__(self, inp1_size, inp2_size, inp3_size, dropout):
        super(Gate, self).__init__()

        self.fc1 = nn.Linear(inp1_size + inp2_size, 1)
        self.fc2 = nn.Linear(inp1_size + inp3_size, 1)
        self.fc3 = nn.Linear(inp2_size + inp3_size, inp1_size)
        self.beta = nn.Parameter(torch.randn((1,)))
        self.norm = nn.LayerNorm(inp1_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp1, inp2, inp3):
        w2 = torch.sigmoid(self.fc1(torch.cat([inp1, inp2], -1)))
        w3 = torch.sigmoid(self.fc2(torch.cat([inp1, inp3], -1)))
        adjust = self.fc3(torch.cat([w2 * inp2, w3 * inp3], -1))
        one = torch.tensor(1).type_as(adjust)
        alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
        output = inp1 + alpha * adjust
        output = self.dropout(self.norm(output))
        return output

class MBertLstm(nn.Module):
    def __init__(self,
                 pretrained_bert: str = 'bvanaken/CORe-clinical-outcome-biobert-v1',
                 ti_input_size: int = 42,
                 ti_norm_size: int = 32,
                 ts_input_size: int = 76,
                 ts_norm_size: int = 64,
                 n_neurons: int = 64,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 att_out: bool = False) -> None:
        super().__init__()
        print('setting up the class')

        self.s_enc = nn.Linear(ti_input_size, ti_norm_size)

        self.X_enc = Lstm(input_size=ts_input_size, hidden_size=ts_norm_size,
                                n_neurons=n_neurons, num_layers=num_layers, dropout=dropout)

        self.nt_enc = Bert(pretrained_bert=pretrained_bert, att_out=att_out)

        self.gate = Gate(bert_size, ti_norm_size, n_neurons, dropout)

        self.pred = nn.Linear(bert_size, output_size)

    def forward(self, input_ids, s, X, token_type_ids, attention_mask):
        s = self.s_enc(s)
        X = self.X_enc(X)

        if self.nt_enc.att_out:
            nt, hidden, att = self.nt_enc(input_ids, token_type_ids, attention_mask)
        else:
            nt = self.nt_enc(input_ids, token_type_ids, attention_mask)


        fusion = self.gate(nt, s, X)

        if self.nt_enc.att_out:
            return torch.sigmoid(self.pred(fusion)).squeeze(1), hidden, att
        else:
            return torch.sigmoid(self.pred(fusion)).squeeze(1)

class MLstmBert(nn.Module):
    def __init__(self,
                 pretrained_bert: str = 'bert-base-uncased',
                 ti_input_size: int = 42,
                 ti_norm_size: int = 32,
                 ts_input_size: int = 76,
                 ts_norm_size: int = 64,
                 n_neurons: int = 64,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 att_out: bool = False) -> None:
        super().__init__()

        self.s_enc = nn.Linear(ti_input_size, ti_norm_size)

        self.X_enc = Lstm(input_size=ts_input_size, hidden_size=ts_norm_size,
                                n_neurons=n_neurons, num_layers=num_layers, dropout=dropout)

        self.nt_enc = Bert(pretrained_bert=pretrained_bert, att_out=att_out)

        self.gate = Gate(n_neurons, ti_norm_size, bert_size, dropout)

        self.pred = nn.Linear(n_neurons, output_size)

    def forward(self, input_ids, s, X, token_type_ids, attention_mask):
        s = self.s_enc(s)
        X = self.X_enc(X)

        if self.nt_enc.att_out:
            nt, hidden, att = self.nt_enc(input_ids, token_type_ids, attention_mask)
        else:
            nt = self.nt_enc(input_ids, token_type_ids, attention_mask)

        fusion = self.gate(X, s, nt)

        if self.nt_enc.att_out:
            return torch.sigmoid(self.pred(fusion)).squeeze(1), hidden, att
        else:
            return torch.sigmoid(self.pred(fusion)).squeeze(1)

class CLstmBert(nn.Module):
    def __init__(self,
                 pretrained_bert: str = 'bvanaken/CORe-clinical-outcome-biobert-v1',
                 ti_input_size: int = 42,
                 ti_norm_size: int = 32,
                 ts_input_size: int = 76,
                 ts_norm_size: int = 76,
                 n_neurons: int = 64,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 att_out: bool = False) -> None:
        super().__init__()

        self.ti_enc = nn.Linear(ti_input_size, ti_norm_size)

        self.ts_enc = Lstm(input_size=ts_input_size, hidden_size=ts_norm_size,
                                n_neurons=n_neurons, num_layers=num_layers, dropout=dropout)

        self.nt_enc = Bert(pretrained_bert=pretrained_bert, att_out=att_out)

        self.pred = nn.Linear(n_neurons + ti_norm_size + bert_size, output_size)

    def forward(self, input_ids, s, X, token_type_ids, attention_mask):
        ti = self.ti_enc(s)
        ts = self.ts_enc(X)

        if self.nt_enc.att_out:
            nt, hidden, att = self.nt_enc(input_ids, token_type_ids, attention_mask)
        else:
            nt = self.nt_enc(input_ids, token_type_ids, attention_mask)

        fusion = torch.cat((ti.float(), ts.float(), nt.float()), 1)

        if self.nt_enc.att_out:
            return torch.sigmoid(self.pred(fusion)).squeeze(1), hidden, att
        else:
            return torch.sigmoid(self.pred(fusion)).squeeze(1)



import yaml
import sys
sys.path.append('/workspace/XAI/checkpoints/MBertLstm_notes/epoch_012.ckpt')
experiment_path = '/workspace/XAI/configs/experiment' # change this PATH to experiment accordingly

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from tqdm.auto import tqdm
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from captum.attr import DeepLiftShap


# as per captum
torch.backends.cudnn.enabled=False

# for the test we are only using one model
configs = ['multi_BL_notes']
loops = ['MBertLstm_notes']

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

total_pred = {}
for l, c  in zip(loops,configs):
    model_name, dataset = l.split('_')

    # load the config
    with open(f'{experiment_path}/{c}.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model_args = config['model']['net']
    check_point = config['ckpt_path'] # change the PATH inside the experiment/multi_BL_notes.yaml under ckpt_path to checkpoints/MBertLstm_notes/epoch_012.ckpt

    # initalize model class and load to device
    model_class = globals().get(model_name)
    model = model_class(**model_args)
    print(model)

    model.load_state_dict(torch.load(check_point,map_location=torch.device('cpu')))

    model.to(device)
    model.eval()

    # initalize the data loader
    dm = MIMICDataModule(notes= True if dataset == 'notes' else False,
                         discharge= True if dataset == 'discharge' else False,
                         time_series=True,
                         batch_size=1)
    dm.prepare_data()
    dm.setup()

    # getting the tokenizer
    if dataset == 'discharge':
        tokenizer = AutoTokenizer.from_pretrained("bvanaken/CORe-clinical-outcome-biobert-v1")
        cls = tokenizer.cls_token_id
        sep = tokenizer.sep_token_id
        msk = tokenizer.mask_token_id
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cls = tokenizer.cls_token_id
        sep = tokenizer.sep_token_id
        msk = tokenizer.mask_token_id

    # wrapping the model to generate attributions
    class EmbeddingInputWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, embeddings, s, X, token_type_ids, attention_mask):
            # Inject embeddings into the BERT encoder manually (no nt_enc call!)
            outputs = self.model.nt_enc.bert(
                inputs_embeds=embeddings,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            nt = outputs.pooler_output

            # Encode other modalities
            s_encoded = self.model.s_enc(s)
            X_encoded = self.model.X_enc(X)

            # Do fusion & prediction (skip self.model.nt_enc())
            fusion = self.model.gate(nt, s_encoded, X_encoded)
            return torch.sigmoid(self.model.pred(fusion)).squeeze(1)


    #import shap
    class EmbeddingInputWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, embeddings, s, X, token_type_ids, attention_mask):
            # Inject embeddings into the BERT encoder manually (no nt_enc call!)
            outputs = self.model.nt_enc.bert(
                inputs_embeds=embeddings,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            nt = outputs.pooler_output

            # Encode other modalities
            s_encoded = self.model.s_enc(s)
            X_encoded = self.model.X_enc(X)

            # Do fusion & prediction (skip self.model.nt_enc())
            fusion = self.model.gate(nt, s_encoded, X_encoded)
            return torch.sigmoid(self.model.pred(fusion)).squeeze(1)
    
    # 1. Wrapper stays the same (EmbeddingInputWrapper)
    wrapped_model = EmbeddingInputWrapper(model)

    # 2. Create the SHAP explainer
    explainer = DeepLiftShap(wrapped_model)

    # 3. Output file
    file_out = open('SHAP_attribution_output_' + model_name + '.jsonl', 'w+') #CHANGE PATH IF NEEDED

    # 4. Loop over test data
    for input_ids, si, xi, token_type_ids, attention_mask, _, _ in tqdm(dm.test_dataloader()):

        # initialize relevancies
        relevancies = {}

        # move to device
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        si = si.to(device)
        xi = xi.to(device)
        attention_mask = attention_mask.to(device)

        # ---- Embed the actual input tokens ----
        embeddings = model.nt_enc.bert.embeddings(input_ids=input_ids)

        # ---- MULTIPLE baselines (SHAP requires multiple baselines) ----
        base_reference_texts = []
        nbr_samples=8
        for _ in range(nbr_samples):  #8 samples
            reference_index = torch.argmax((input_ids == sep).int()).item()
            base_reference_text = [cls] + [msk] * (reference_index - 1) + input_ids[0, reference_index:].tolist()
            base_reference_text = torch.tensor(base_reference_text, dtype=torch.long).unsqueeze(0).to(device)
            base_embed = model.nt_enc.bert.embeddings(input_ids=base_reference_text)
            base_reference_texts.append(base_embed)

        base_embeds = torch.cat(base_reference_texts, dim=0)  # (nbr_samples, seq_len, hidden_dim)

        # Other modality baselines (batchified for SHAP)
        base_time_series = ((0 * xi) + -1).repeat(nbr_samples, 1, 1).to(device)  # (nbr_samples, seq_len, features)
        base_time_invarient = ((0 * si) + -1).repeat(nbr_samples, 1).to(device)  # (nbr_samples, features)

        # ---- Setup inputs and baselines ----
        inputs = (embeddings, si, xi)
        baselines = (base_embeds, base_time_invarient, base_time_series)
        add_args = (token_type_ids, attention_mask)

        # ---- Run SHAP ----
        attribution = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            additional_forward_args=add_args
        )

        # ---- Extract attributions ----
        attribution_txt = attribution[0].squeeze()
        attribution_s = attribution[1].squeeze()
        attribution_X = attribution[2].squeeze()

        relevancies['attri_s'] = attribution_s.cpu().detach().numpy().tolist()
        relevancies['attri_X'] = attribution_X.cpu().detach().numpy().tolist()
        relevancies['attri_txt_norm'] = torch.norm(attribution_txt, dim=-1).cpu().detach().numpy().tolist()
        relevancies['attri_txt_mean'] = torch.mean(attribution_txt, dim=-1).cpu().detach().numpy().tolist()

        # dump the relevancies
        file_out.write(json.dumps(relevancies) + '\n')

    file_out.close()
