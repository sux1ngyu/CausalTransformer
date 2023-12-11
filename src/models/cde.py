import torch
import torchcde
import torchdiffeq
from torch.nn import functional as F
import logging
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import numpy as np
from typing import Union

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models import TimeVaryingCausalModel, BRCausalModel
from src.models.utils import BRTreatmentOutcomeHead
from src.models.utils_cde import CDEFunc

from tqdm import tqdm

logger = logging.getLogger(__name__)


class NeuralCDE(BRCausalModel):
    model_type = 'multi'
    possible_model_types = {'multi'}
    tuning_criterion = 'rmse'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None, **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon

        # Used in hparam tuning
        self.input_size = self.dim_treatments + self.dim_static_features + self.dim_vitals + self.dim_outcome
        logger.info(f'The input size of {self.model_type}: {self.input_size}')
        assert self.autoregressive  # prev_outcomes are obligatory

        self._init_specific(args.model.multi)
        self.save_hyperparameters(args)

    def _init_specific(self, sub_args: DictConfig):
        """
        Initialization of network
        """
        try:
            self.dropout_rate = sub_args.dropout_rate
            self.hidden_units = sub_args.hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.br_size = sub_args.br_size

            # input layer
            self.embed_x = torch.nn.Linear(self.input_size, self.hidden_units)

            # model of derivation f
            self.net = CDEFunc(self.input_size, self.hidden_units)
            # linear output layer
            self.outcome = torch.nn.Linear(self.hidden_units, self.dim_outcome)  # 8 -> 59
            self.treatment = torch.nn.Linear(self.hidden_units, self.dim_treatments)  # 8 -> 4

            self.dropout_layer = torch.nn.Dropout(self.dropout_rate)

            # treatment and outcome output layer
            self.br_treatment_outcome_head = BRTreatmentOutcomeHead(self.hidden_units, self.br_size,
                                                                    self.fc_hidden_units, self.dim_treatments, self.dim_outcome,
                                                                    self.alpha, self.update_alpha, self.balancing)

        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    def forward(self, batch, detach_treatment=False):
        fixed_split = batch['future_past_split'] if 'future_past_split' in batch else None

        if self.training and self.hparams.model.multi.augment_with_masked_vitals and self.has_vitals:
            # Augmenting original batch with vitals-masked copy
            assert fixed_split is None  # Only for training data
            fixed_split = torch.empty((2 * len(batch['active_entries']),)).type_as(batch['active_entries'])
            for i, seq_len in enumerate(batch['active_entries'].sum(1).int()):
                fixed_split[i] = seq_len  # Original batch
                fixed_split[len(batch['active_entries']) + i] = torch.randint(0, int(seq_len) + 1, (1,)).item()  # Augmented batch

            for (k, v) in batch.items():
                batch[k] = torch.cat((v, v), dim=0)

        prev_treatments = batch['prev_treatments']  # treatments :-1
        vitals = batch['vitals'] if self.has_vitals else None  # vitals 1:
        prev_outputs = batch['prev_outputs']  # outcomes :-1
        static_features = batch['static_features']  # static_features, need unsqueeze(1) on time
        curr_treatments = batch['current_treatments']  # treatments 1:
        # only one time ahead of the prev_treatments
        active_entries = batch['active_entries']  # active_entries 1:

        # In synthetic dataset, we want the covariates and treatments to be :-1. outcome 1: (covariates contain outomes)
        # In real dataset, we want previous treatment, current covariates, previous outputs to predict current output & current treatment
        observ_x = torch.concat((prev_treatments, vitals, prev_outputs, static_features.unsqueeze(1).repeat(1, vitals.shape[1], 1)), dim=2)

        batch_size, sequence_len, _ = observ_x.shape

        z0_value = self.embed_x(observ_x[:, 0, :])
        # 512 * 144

        pred_series = []
        for time_step in range(sequence_len):
            if time_step == 0:
                # first point: ODE
                time_interval = torch.arange(time_step, time_step+2, dtype=torch.float, device=z0_value.device)
                self.net.ode_case = True
                z_t = torchdiffeq.odeint_adjoint(func=self.net, y0=z0_value, t=time_interval)[-1]
                # output both begin point and end point
            else:
                coeffs_x = torchcde.linear_interpolation_coeffs(observ_x[:, :(time_step+1), :])
                x = torchcde.LinearInterpolation(coeffs_x)
                self.net.ode_case = False
                z_t = torchcde.cdeint(X=x, z0=z0_value, func=self.net, t=x.interval, backend="torchdiffeq", method="dopri5")[:, 1]
            pred_series.append(z_t)
        pred_series = torch.stack(pred_series, dim=1)

        pred_series = self.dropout_layer(pred_series)

        # output layers
        br = self.br_treatment_outcome_head.build_br(pred_series)
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)  # # bn*seq*dim
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)  # bn*seq*dim

        return treatment_pred, outcome_pred, br


