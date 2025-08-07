import darts
from darts.models import NHiTSModel


class Optimization:
    """
    End-to-end trigger-reconstruction helper for the “Trojan Horse Hunt” baseline.

    The class wraps a pre-trained **N-HiTS** model and a clean validation snippet,
    then offers utilities to:

    1. **Probe** the model with synthetic spikes and infer which telemetry channels
       are trojaned.
    2. **Optimise** an additive time series trigger (one per poisoned channel) that
       maximises forecast deviation while remaining time-local.
    3. **Inspect** optimisation logs, smooth / post-process the trigger, convert it
       into the competition’s `(3, 75)` format, and plot qualitative diagnostics.

    Parameters
    ----------
    model : NHiTSModel
        A fitted darts.models.NHiTSModel that may contain a trojan.
    val_clean : TimeSeries
        Clean validation slice used for probing and as the optimisation baseline.
    insert_pos : int, default=200
        Index at which the trigger is inserted into the input series.
    trigger_duration : int, default=75
        Length (in timesteps) of the additive trigger.
    lambda_reg : float, default=0.5
        Weight on the ℓ² regularisation term of the trigger (`||δ||₂`).
    alpha_reg : float, default=1.5
        Weight on the *tracking* loss (forces forecast to follow the trigger).
    beta_reg : float, default=2
        Weight on the *difference* loss (distance between poisoned and clean
        forecasts).
    epochs : int, default=100
        Number of optimisation steps per channel.
    forecast_horizon : int, default=400
        Length of the model forecast used in the loss computations.
    input_chunk_length : int, default=400
        Number of historical points fed to the model.
    """
    def __init__(self, model: NHiTSModel,
                 val_clean: TimeSeries,
                 insert_pos: int = 200,
                 trigger_duration = 75,
                 lambda_reg: float = 0.5,
                 alpha_reg: float = 1.5,
                 beta_reg: float = 2,
                 epochs: int = 100,
                 forecast_horizon: int = 400,
                 input_chunk_length: int = 400):
        
        self.model = model
        self.val_clean = val_clean
        self.insert_pos = insert_pos
        self.trigger_duration = trigger_duration
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg
        self.beta_reg = beta_reg
        self.epochs = epochs
        self.forecast_horizon = forecast_horizon
        self.input_chunk_length = input_chunk_length
        self.channels = ["channel_44", "channel_45", "channel_46"]


    def get_poisoned_channels(self, spike_value: float):
        """
        Heuristically identify which of the three telemetry channels are trojaned.

        A short spike (`spike_value`) is injected into each candidate channel.
        If the resulting forecast exceeds a channel-specific min/max tolerance
        band, that channel is flagged as *poisoned*.

        Parameters
        ----------
        spike_value : float
            Amplitude of the synthetic spike inserted between indices 250–259.

        Returns
        -------
        list[str]
            Channel names (e.g. ``["channel_44", "channel_46"]``) whose forecasts
            react anomalously and are therefore suspected to carry a back-door.
        """

        val_probed_df_copy = self.val_clean[:self.input_chunk_length].pd_dataframe().copy()  # Create a copy of the DataFrame
        for channel in self.channels:
            val_probed_df_copy[channel].iloc[250:260] = spike_value
        val_probed_df_copy = TimeSeries.from_dataframe(val_probed_df_copy.astype("float32"))

        val_clean_df_copy = self.val_clean[:self.input_chunk_length].pd_dataframe().copy()
        stat = val_clean_df_copy.describe()
        
        poisoned_channels = []

        for channel in self.channels:
            more_than_max = (stat[channel]["max"] + 0.5*(stat[channel]["max"] - stat[channel]["min"])) > self.model.predict(400, series=val_probed_df_copy, num_samples=1)[channel].values()
            any_more_than_max = not more_than_max.all()
            
            less_than_min = (stat[channel]["min"] - 0.5*(stat[channel]["max"] - stat[channel]["min"])) < self.model.predict(400, series=val_probed_df_copy, num_samples=1)[channel].values()
            any_less_than_min = not less_than_min.all()

            if (any_more_than_max or any_less_than_min):
                poisoned_channels.append(channel)
        return poisoned_channels



    def get_num_poisoned_channels(self, poisoned_channels: list[str]):
        """
        Map poisoned channel names to their positional indices (0, 1, 2).

        Parameters
        ----------
        poisoned_channels : list[str]
            Output of :py:meth:`get_poisoned_channels`.

        Returns
        -------
        list[int]
            Zero-based indices corresponding to ``val_clean.components``.
        """
        """
        Returns a list of positions (0-based indices) of poisoned channels in val_poisoned.
        """
        val_poisoned_channels = list(self.val_clean.components)
        return [val_poisoned_channels.index(ch) for ch in poisoned_channels if ch in val_poisoned_channels]

    
    def create_input_tensor(self):
        """
        Convert the first ``input_chunk_length`` timesteps of ``val_clean`` into
        a Torch tensor of shape ``(T, 3)``.

        Returns
        -------
        torch.Tensor
            Clean input ready for in-place trigger injection.
        """
        clean_input_np = self.val_clean[:self.input_chunk_length].values()
        input_tensor = torch.tensor(clean_input_np, dtype=torch.float32)
        return input_tensor
    
    def create_clean_input(self):
        """
        Convenience wrapper that clones the clean input tensor and converts it
        back to a :class:`darts.TimeSeries`. Mainly used when a detached copy of
        the untouched sequence is required.
        """
        clean = self.create_input_tensor().clone()
        clean = TimeSeries.from_values(clean.detach().numpy())

    def discover_trigger_injection(self, 
                                   input_tensor: torch.Tensor, 
                                   channel: int, 
                                   epochs: int=200):
        """
        Gradient-search for an additive trigger on a single channel.

        A learnable vector ``δ ∈ ℝ^{trigger_duration}`` is inserted at
        ``insert_pos`` on the specified channel and optimised with an AdamW
        + StepLR schedule.

        Loss
        ----
        ``loss = -β·diff_loss  - λ·‖δ‖₂  + α·tracking_loss``

        * **diff_loss** – L1 distance between poisoned and clean forecasts  
        * **tracking_loss** – L1 distance between poisoned forecast and the
          trigger values themselves (encourages the forecast to “follow” δ)  
        * **‖δ‖₂** – ℓ² size penalty

        Parameters
        ----------
        input_tensor : torch.Tensor
            Clean input series on which to graft the trigger.
        channel : int
            Index (0, 1, 2) of the candidate poisoned channel.
        epochs : int, default=200
            Optimisation iterations.

        Returns
        -------
        tuple[np.ndarray, TimeSeries, pandas.DataFrame]
            1. Optimised trigger δ (shape ``(trigger_duration,)``)  
            2. Modified input series with δ applied (TimeSeries)  
            3. Per-epoch log of ``tracking_loss``, ``diff_loss``, ``reg_loss``
        """
  
        delta = torch.zeros((self.trigger_duration, 1), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.AdamW([delta], lr=0.2, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

        for i in tqdm(range(epochs)):

            optimizer.zero_grad()
            modified = input_tensor.clone()
            modified[self.insert_pos:self.insert_pos + self.trigger_duration, channel] += delta.squeeze()
            modified_series = TimeSeries.from_values(modified.detach().numpy())

            
            pred_poisoned = self.model.predict(n=self.forecast_horizon, series=modified_series, verbose=False)
            pred_clean = self.model.predict(n=self.forecast_horizon, series=self.val_clean[:self.input_chunk_length], verbose=False)
            poisoned_tensor = torch.tensor(pred_poisoned.values(), dtype=torch.float32)
            clean_tensor = torch.tensor(pred_clean.values(), dtype=torch.float32)

            # for guess_channel in guess_poisoned_channel:
            tracking_target = modified[-self.forecast_horizon:, channel] 

            tracking_loss = torch.sum(torch.abs(poisoned_tensor[:, channel] - tracking_target))
            diff_loss = torch.sum(torch.abs(poisoned_tensor[:, channel] - clean_tensor[:, channel]))
            reg_loss =  torch.norm(delta, p=2)
            
            loss = - self.beta_reg * diff_loss \
                - self.lambda_reg * reg_loss \
                + self.alpha_reg * tracking_loss
                
            discovered_trigger = delta.detach().numpy().flatten()
        
            # Log metrics to a DataFrame after each iteration
            if 'opt_log_df' not in locals():
                opt_log_df = pd.DataFrame(columns=["epoch", "tracking_loss", "diff_loss", "reg_loss"])
            opt_log_df.loc[len(opt_log_df)] = [i, tracking_loss.item(), diff_loss.item(), reg_loss.item()]

            if i != epochs - 1:
                loss.backward()
                optimizer.step()
                scheduler.step()

        modified = input_tensor.clone()
        modified[self.insert_pos:self.insert_pos + self.trigger_duration, channel] += delta.squeeze()
        modified_series = TimeSeries.from_values(modified.detach().numpy())
        
        return delta.detach().numpy().flatten(), modified_series, opt_log_df
    
    def find_best_trigger(self, opt_log_df: pd.DataFrame,
                          poisoned_channels: list[str],
                          target_preds_diff: float = 2.5,
                          target_context_pred_diff: float = 4,
                          target_reg: float = 0.075):
        """
        Rank optimisation epochs by distance to hand-tuned target loss values.

        A Euclidean distance is computed in the 3-D space
        ``(diff_loss, tracking_loss, reg_loss)`` with a ×10 weight on reg_loss
        to keep the trigger small.

        Returns
        -------
        pandas.DataFrame
            ``opt_log_df`` sorted ascending by the new column ``loss_distance``.
        """

        
        ## global targets used for finding the best trigger
        opt_log_df['loss_distance'] = np.sqrt((opt_log_df['diff_loss'] - target_preds_diff)**2 \
                                    + (opt_log_df['tracking_loss'] - target_context_pred_diff)**2 \
                                    + 10*(opt_log_df['reg_loss'] - target_reg)**2)
        
        opt_log_df.sort_values(by=['loss_distance'], inplace=True)
        return opt_log_df
    
    def smooth_discovered_trigger(self, 
                                  discovered_triggers: dict, 
                                  poisoned_channel: list[str]):
        """
        Apply a Savitzky-Golay filter (window=15, polyorder=3) to reduce noise in
        the raw δ estimates for each poisoned channel.

        Returns
        -------
        dict
            Same structure as ``discovered_triggers`` with smoothed arrays.
        """
        for channel in self.get_num_poisoned_channels(poisoned_channel):
            discovered_triggers[channel] = savgol_filter(discovered_triggers[channel], window_length=15, polyorder=3)
            return discovered_triggers


    def transform_discovered_trigger(self, 
                                     discovered_triggers: dict, 
                                     poisoned_channels: list[str]):
        """
        Convert per-channel trigger dict into a `(3, trigger_duration)` numpy
        array with zeros for non-poisoned channels—matching the competition’s
        submission format.

        Returns
        -------
        np.ndarray
            Final trigger tensor ready for saving or plotting.
        """
        zero_trigger = np.zeros(self.trigger_duration)
        zero_trigger = zero_trigger.astype(np.float32)
        trigger = [zero_trigger, zero_trigger, zero_trigger]

        for channel in self.get_num_poisoned_channels(poisoned_channels):
            trigger[channel] = discovered_triggers[channel]
        discovered_trigger = np.array(trigger)
        discovered_trigger = discovered_trigger.astype(np.float32)
        return discovered_trigger


    def plot_discovered_trigger(self, discovered_trigger: np.ndarray):
        """
        Visualise the reconstructed trigger overlayed on a zero baseline.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing three stacked sub-plots (one per channel).
        """
        
        zero_trigger = np.zeros(self.trigger_duration)
        zero_trigger = zero_trigger.astype(np.float32)
        fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True)
        for i in range(3):
            # axs[i].plot(self.trigger[i], label="GT Trigger", color="blue")
            axs[i].plot(discovered_trigger[i], label="Discovered Trigger", color="black")
            axs[i].plot(zero_trigger, color="red", linestyle="--", label="Zero Trigger")
            axs[i].set_ylabel(f'Channel {i+44} Amplitude')
            axs[i].set_title('Discovered Trigger' + f' - Channel {i+44}')

            axs[i].legend(
                [
                # "Original Trigger 0 error",
                "Discovered Trigger", 
                "Zero Trigger"
                ],
                loc="lower center",
                ncol=1,
                fontsize=12,
                frameon=True,
                bbox_to_anchor=(0.4, 0.02)
            )
        plt.xlabel("Trigger Duration")
        plt.grid()
        return fig


    def plot_triggered_model(self, 
                             modified_series: dict,
                             poisoned_channels: list[str]):
        """
        Compare clean vs poisoned forecasts alongside the ground-truth series.

        For each poisoned channel the function plots:

        * Original clean input  
        * Clean forecast  
        * Triggered input (if applicable)  
        * Poisoned forecast (if applicable)

        Returns
        -------
        matplotlib.figure.Figure
            Diagnostic plot for qualitative assessment of trigger effect.
        """

        time_index = self.val_clean[:self.input_chunk_length].time_index
        for channel in self.get_num_poisoned_channels(poisoned_channels):
            modified_series[channel] = TimeSeries.from_times_and_values(
                times=time_index,
                values=modified_series[channel].values(),
                columns=["0", "1", "2"]
            )

        pred_clean = self.model.predict(self.forecast_horizon, self.val_clean[:self.input_chunk_length])
        


        fig = plt.figure(figsize=(12, 6))

        self.val_clean[:self.input_chunk_length]['channel_44'].plot(label="Actual channel_44", color="green")
        pred_clean['channel_44'].plot(color="grey", label="Forecast clean channel_44")
        if "channel_44" in poisoned_channels:
            pred_poisoned = self.model.predict(self.forecast_horizon, modified_series[0])
            pred_poisoned['0'].plot(label="Forecast poisoned channel_44", color="salmon")
            modified_series[0]['0'].plot(label="Actual channel_44", color="black")

        
        self.val_clean[:self.input_chunk_length]['channel_45'].plot(label="Actual channel_45", color="green")
        pred_clean['channel_45'].plot(label="Forecast channel_45", color="lightblue")
        if "channel_45" in poisoned_channels:
            pred_poisoned = self.model.predict(self.forecast_horizon, modified_series[1])
            pred_poisoned['1'].plot(label="Forecast poisoned channel_45", color="salmon")
            modified_series[1]['1'].plot(label="Actual channel_45", color="blue")

        self.val_clean[:self.input_chunk_length]['channel_46'].plot(label="Actual channel_45", color="green")
        pred_clean['channel_46'].plot(label="Forecast channel_46", color="lightgreen")
        if "channel_46" in poisoned_channels:
            pred_poisoned = self.model.predict(self.forecast_horizon, modified_series[2])
            pred_poisoned['2'].plot(label="Forecast poisoned channel_46", color="salmon")
            modified_series[2]['2'].plot(label="Triggered channel_46", color="red")
        

        plt.legend()

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.xlabel("Time", fontsize=14)
        plt.grid()
        legend = plt.legend(fontsize=12, frameon=True)
        legend.get_frame().set_alpha(0.7)
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1)
        plt.title("Triggered Model Evaluation", fontsize=16)

        plt.legend().set_visible(False)

        return fig
