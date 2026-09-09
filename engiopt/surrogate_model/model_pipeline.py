"""Utility classes for data preprocessing and model pipeline management.

Handles raw data transformations for both training and inference. This includes
stripping spaces, flattening list columns, optional subset filtering, nondimensionalization,
log-transformation, and splitting parameter columns into continuous and categorical parts.
For columns with fewer than 5 unique values, one-hot encoding is applied. The preprocessor
stores the final parameter column names and the list of dummy columns for categorical columns.

Usage in training:
  preprocessor = DataPreprocessor(vars(args))
  processed_dict = preprocessor.transform_inputs(df, fit_params=True)
  -> Returns a dict with keys:
       If structured: {"x_init": ..., "x_opt": ..., "params": ...}
       Else: {"X": ...}
  It also stores self.final_param_columns, self.continuous_columns,
  self.categorical_columns, and self.categorical_mapping.

Usage in inference:
  (After loading the pipeline, the pipeline calls preprocessor.transform_inputs(raw_df, fit_params=False)
  so that the same dummy columns are created.)
"""

from __future__ import annotations

import ast
import math
import os
import pickle
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

MIN_UNIQUE_FOR_ONEHOT = 5


class DataPreprocessor:
    """Handles raw data transformations.

    Flattens list columns, applies optional subset filtering,
    nondimensionalization, log-transformations, and processes parameter columns.
    """

    def __init__(self, args: dict[str, Any]) -> None:
        """Initialize the DataPreprocessor.

        Args:
            args: Dictionary of configuration arguments.
        """
        self.args = args.copy()
        self.final_param_columns: list[str] | None = None
        self.continuous_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.categorical_mapping: dict[str, list[str]] = {}

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input DataFrame.

        Strips column spaces, flattens columns, applies subset filtering,
        nondimensionalization, and log-transforms the target if requested.

        Args:
            df: Input DataFrame to preprocess.

        Returns:
            Preprocessed DataFrame.
        """
        # 1) Flatten columns if requested.
        flatten_cols = self.args.get("flatten_columns", [])
        if flatten_cols:
            df = self._flatten_list_columns(df, flatten_cols)

        # 2) Subset condition.
        subset_condition = self.args.get("subset_condition", None)
        if subset_condition:
            df = df.query(subset_condition)

        # 3) Nondimensionalization.
        nondim_map = self.args.get("nondim_map", None)
        if nondim_map:
            if isinstance(nondim_map, str):
                nondim_map = ast.literal_eval(nondim_map)
            for col, ref in nondim_map.items():
                if col in df.columns and ref in df.columns:
                    df[col] = df[col] / df[ref]

        # 4) Log-transform target if requested.
        if self.args.get("log_target", False):
            target_col = self.args.get("target_col", None)
            if target_col in df.columns:
                df[target_col] = np.log(df[target_col])
                print(f"[DataPreprocessor] Applied log-transform to {target_col}")

        return df

    def transform_inputs(
        self,
        df: pd.DataFrame,
        *,
        fit_params: bool = False,
    ) -> tuple[dict[str, npt.NDArray], pd.DataFrame]:
        """Transforms raw data into arrays for model input.

        Returns both the input arrays and the processed DataFrame.

        In structured mode (e.g. VAE with shapes):
          Returns ({"x_init": ..., "x_opt": ..., "params": ...}, df_processed).

        In unstructured mode (e.g. plain MLP):
          Returns ({"X": ...}, df_processed).

        When fit_params is True, one-hot/dummy logic is applied to parameter columns.
        When False, stored mappings from a previous run are used.

        Args:
            df: The raw DataFrame to be transformed.
            fit_params: If True, fits new parameter mappings (e.g., one-hot encoding).
                        If False, uses stored mappings.

        Returns:
            A tuple containing:
              - A dictionary of NumPy arrays containing model inputs.
                * In structured mode: {"x_init", "x_opt", "params"}.
                * In unstructured mode: {"X"}.
              - The processed DataFrame.
        """
        df_processed = self.preprocess_dataframe(df.copy())
        structured = self.args.get("structured", False)
        init_col = self.args.get("init_col", "")
        opt_col = self.args.get("opt_col", "")
        param_cols = self.args.get("params_cols", [])

        if structured:
            # Gather shape columns
            init_cols = [c for c in df_processed.columns if c.startswith(init_col + "_")]
            opt_cols = [c for c in df_processed.columns if c.startswith(opt_col + "_")]
            x_init = df_processed[init_cols].to_numpy() if init_cols else None
            x_opt = df_processed[opt_cols].to_numpy() if opt_cols else None

            # Process parameter columns
            if fit_params:
                param_df = self._split_params_with_onehot(df_processed, param_cols)
            else:
                param_df = self._apply_params_inference(df_processed, param_cols)
            params = param_df.to_numpy() if len(param_df.columns) > 0 else np.empty((len(df_processed), 0))
            return {"x_init": x_init, "x_opt": x_opt, "params": params}, df_processed  # type: ignore  # noqa: PGH003
        # Unstructured mode
        if fit_params:
            param_df = self._split_params_with_onehot(df_processed, param_cols)
        else:
            param_df = self._apply_params_inference(df_processed, param_cols)
        x = param_df.to_numpy() if len(param_df.columns) > 0 else np.empty((len(df_processed), 0))
        return {"X": x}, df_processed

    def _split_params_with_onehot(self, df: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
        """Splits parameter columns into continuous and categorical subsets.

        Applies one-hot encoding to categorical columns with fewer than five unique values.

        This method is called during fitting (fit_params=True) and updates internal
        attributes such as continuous_columns, categorical_columns, and categorical_mapping.

        Args:
            df: The preprocessed DataFrame containing parameter columns.
            param_cols: List of columns to split.

        Raises:
            ValueError: If a specified parameter column is not found in df.

        Returns:
            A new DataFrame with continuous columns followed by one-hot-encoded columns.
        """
        if not param_cols:
            return pd.DataFrame(index=df.index)
        cont_list = []
        cat_list = []
        self.continuous_columns = []
        self.categorical_columns = []
        self.categorical_mapping = {}
        for col in param_cols:
            if col not in df.columns:
                raise ValueError(f"Parameter column '{col}' not found.")
            if df[col].nunique() < MIN_UNIQUE_FOR_ONEHOT:
                dummies = pd.get_dummies(df[col], prefix=col)
                cat_list.append(dummies)
                self.categorical_columns.append(col)
                self.categorical_mapping[col] = list(dummies.columns)
            else:
                cont_list.append(df[[col]])
                self.continuous_columns.append(col)
        cont_df = pd.concat(cont_list, axis=1) if cont_list else pd.DataFrame(index=df.index)
        cat_df = pd.concat(cat_list, axis=1) if cat_list else pd.DataFrame(index=df.index)
        final_df = pd.concat([cont_df, cat_df], axis=1)
        self.final_param_columns = list(final_df.columns)
        return final_df

    def _apply_params_inference(self, df: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
        """Applies parameter splitting logic in inference mode.

        Uses stored continuous/categorical mappings to reconstruct the same dummy columns as during training.
        For continuous columns, copies them directly if they exist or creates columns of NaNs if missing.
        For categorical columns, re-creates dummies and reindexes them to match the training dummy columns.

        Args:
            df: DataFrame containing parameter columns for inference.
            param_cols: List of columns to process.

        Returns:
            The transformed DataFrame with continuous and dummy columns in the same order as during training.
        """
        if not param_cols:
            return pd.DataFrame(index=df.index)
        # For continuous columns: simply get the column.
        cont_dfs = []
        for col in self.continuous_columns:
            if col in df.columns:
                cont_dfs.append(df[[col]])
            else:
                cont_dfs.append(pd.DataFrame(np.nan, index=df.index, columns=[col]))
        cont_df = pd.concat(cont_dfs, axis=1) if cont_dfs else pd.DataFrame(index=df.index)

        # For categorical columns: re-create dummies and reindex with stored dummy columns.
        cat_dfs = []
        for col in self.categorical_columns:
            if col not in df.columns:
                dummy_cols = self.categorical_mapping.get(col, [])
                cat_dfs.append(pd.DataFrame(0, index=df.index, columns=dummy_cols))
            else:
                dummies = pd.get_dummies(df[col], prefix=col)
                expected_cols = self.categorical_mapping.get(col, [])
                dummies = dummies.reindex(columns=expected_cols, fill_value=0)
                cat_dfs.append(dummies)
        cat_df = pd.concat(cat_dfs, axis=1) if cat_dfs else pd.DataFrame(index=df.index)

        final_df = pd.concat([cont_df, cat_df], axis=1)
        if self.final_param_columns:
            final_df = final_df.reindex(columns=self.final_param_columns, fill_value=0)
        return final_df

    def _flatten_list_columns(self, df: pd.DataFrame, columns_to_flatten: list[str]) -> pd.DataFrame:
        """Flattens specified columns where each entry is a list, tuple, or numpy.ndarray.

        For each flattened column, creates new columns named '{original_col}_{index}'.

        Args:
            df: Input DataFrame.
            columns_to_flatten: List of column names to flatten.

        Returns:
            DataFrame with flattened columns.
        """
        new_cols_list = []
        drop_cols = []

        for col in columns_to_flatten:
            if col not in df.columns:
                continue

            first_val = df[col].iloc[0]

            # If it's a string that might represent a list, try literal_eval.
            if isinstance(first_val, str):
                try:
                    _ = ast.literal_eval(first_val)
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                except (ValueError, SyntaxError) as e:
                    print(f"[WARN] Could not parse '{col}' as string literal: {e}. Proceeding.")

            # If the value is not a list/tuple/ndarray but is iterable (and not a string), cast it.
            if (
                not isinstance(df[col].iloc[0], (list, tuple, np.ndarray))
                and hasattr(df[col].iloc[0], "__iter__")
                and not isinstance(df[col].iloc[0], str)
            ):
                df[col] = df[col].apply(lambda x: list(x) if x is not None else x)

            if isinstance(df[col].iloc[0], np.ndarray):
                df[col] = df[col].apply(lambda x: x.tolist())

            if isinstance(df[col].iloc[0], (list, tuple)):
                flattened_rows = [self._recursive_flatten(x) for x in df[col]]
                lengths = [len(row) for row in flattened_rows]
                if len(set(lengths)) > 1:
                    raise ValueError(f"Varying row lengths in column '{col}': {set(lengths)}")
                n = lengths[0]
                new_col_names = [f"{col}_{i}" for i in range(n)]
                expanded_df = pd.DataFrame(flattened_rows, columns=new_col_names)
                new_cols_list.append(expanded_df)
                drop_cols.append(col)
            else:
                print(f"Column '{col}' is not recognized as list-like; skipping flatten.")

        if new_cols_list:
            df = pd.concat([df.drop(columns=drop_cols).reset_index(drop=True), *new_cols_list], axis=1)
        return df

    def _recursive_flatten(self, val: object) -> list[Any]:
        """Recursively flattens nested lists or tuples.

        Args:
            val: Value to flatten.

        Returns:
            Flattened list.
        """
        if not isinstance(val, (list, tuple)):
            return [val]
        result = []
        for item in val:
            result.extend(self._recursive_flatten(item))
        return result


###############################################################################
# ModelPipeline
###############################################################################
class ModelPipeline:
    """Wraps one or more trained PyTorch models, fitted scalers, and a DataPreprocessor.

    Provides convenience methods for prediction, evaluation, saving, and loading.
    """

    def __init__(
        self,
        models: list[torch.nn.Module],
        scalers: dict[str, Any],
        *,
        structured: bool,
        log_target: bool,
        metadata: dict[str, Any] | None = None,
        preprocessor: DataPreprocessor | None = None,
    ) -> None:
        """Initialize the ModelPipeline.

        Args:
            models: List of PyTorch models.
            scalers: Dictionary of fitted scalers.
            structured: Whether the model is structured (e.g. VAE) or not.
            log_target: Whether the target is log-transformed.
            metadata: Optional metadata dictionary.
            preprocessor: Optional DataPreprocessor instance.
        """
        self.models = models
        self.scalers = scalers
        self.structured = structured
        self.log_target = log_target
        self.metadata = metadata if metadata is not None else {}
        self.preprocessor = preprocessor
        for m in self.models:
            m.eval()

    def to_device(self, device: torch.device | None = None) -> None:
        """Moves all models in the pipeline to the specified device (CPU or GPU).

        Args:
            device: The device to move the models onto. Defaults to CPU if None.
        """
        if device is None:
            device = torch.device("cpu")
        for i, m in enumerate(self.models):
            self.models[i] = m.to(device)

    def predict(
        self,
        raw_input: pd.DataFrame,
        batch_size: int = 256,
        device: torch.device | None = None,
    ) -> npt.NDArray:
        """Predicts target values from raw input data using the stored ensemble of models.

        Applies the stored DataPreprocessor to transform raw_input, scales parameters,
        batches the data, runs forward passes for each model, and averages predictions.
        In structured mode, additional processing of shape arrays is performed.

        Args:
            raw_input: Raw features to predict on.
            batch_size: Batch size for inference.
            device: Device for inference. Defaults to CPU if None.

        Raises:
            ValueError: If no preprocessor is available or required scalers are missing.

        Returns:
            1D array of predictions.
        """
        if device is None:
            device = torch.device("cpu")
        self.to_device(device)

        processed_data = self._process_input_data(raw_input)
        params_scaled = self._scale_parameters(processed_data)
        params_tensor, x_init_tensor = self._prepare_tensors(processed_data, params_scaled)
        predictions = self._run_predictions(params_tensor, x_init_tensor, batch_size, device)
        return self._post_process_predictions(predictions)

    def _process_input_data(self, raw_input: pd.DataFrame) -> dict[str, np.ndarray]:
        """Process raw input data using the preprocessor."""
        if not self.preprocessor:
            raise ValueError("No preprocessor available")
        processed, _ = self.preprocessor.transform_inputs(raw_input, fit_params=False)
        return processed

    # in ModelPipeline._scale_parameters
    def _scale_parameters(self, processed_data):
        params = processed_data["X"] if not self.structured else processed_data["params"]
        if "scaler_params" in self.scalers:
            return self.scalers["scaler_params"].transform(params)
        # allow lowercase key
        if "scaler_x" in self.scalers:
            return self.scalers["scaler_x"].transform(params)
        if "scaler_X" in self.scalers:
            return self.scalers["scaler_X"].transform(params)
        return params

    def _prepare_tensors(
        self, processed_data: dict[str, np.ndarray], params_scaled: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare tensors for model input."""
        params_tensor = torch.from_numpy(params_scaled).float()

        if self.structured:
            x_init = processed_data["x_init"]
            if "scaler_init" not in self.scalers:
                raise ValueError("Missing scaler_init for structured mode")
            x_init_scaled = self.scalers["scaler_init"].transform(x_init)
            x_init_tensor = torch.from_numpy(x_init_scaled).float()
        else:
            x_init_tensor = None

        return params_tensor, x_init_tensor

    def _run_predictions(
        self, params_tensor: torch.Tensor, x_init_tensor: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> npt.NDArray:
        """Run averaged predictions from all models in the ensemble."""
        all_preds = []
        num_samples = params_tensor.shape[0]
        n_batches = math.ceil(num_samples / batch_size)

        for b_idx in range(n_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, num_samples)
            param_batch = params_tensor[start:end].to(device)

            if x_init_tensor is not None:
                x_init_batch = x_init_tensor[start:end].to(device)

            ensemble_preds_list: list[torch.Tensor] = []
            for model in self.models:
                with torch.no_grad():
                    if self.structured:
                        _, _, _, _, cl_pred = model(x_init_batch, param_batch)
                        pred = cl_pred.view(-1)
                    else:
                        pred = model(param_batch).view(-1)
                ensemble_preds_list.append(pred)

            ensemble_preds = torch.stack(ensemble_preds_list, dim=0)
            batch_mean = ensemble_preds.mean(dim=0)
            all_preds.append(batch_mean.cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def _post_process_predictions(self, predictions: npt.NDArray) -> npt.NDArray:
        """Apply post-processing to predictions."""
        if "scaler_y" in self.scalers:
            predictions = self.scalers["scaler_y"].inverse_transform(predictions.reshape(-1, 1)).flatten()
        if self.log_target:
            predictions = np.exp(predictions)
        return predictions

    def evaluate(
        self,
        raw_input: pd.DataFrame,
        y_true: npt.NDArray,
        batch_size: int = 256,
        device: torch.device | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluates predictions against ground-truth using specified metrics.

        Calls predict() and then computes errors such as MSE, RMSE, MAE, and relative error.

        Args:
            raw_input: Raw features for evaluation.
            y_true: Ground-truth target values.
            batch_size: Batch size for prediction.
            device: Device for inference. Defaults to CPU if None.
            metrics: List of metrics to compute (e.g., "mse", "rmse", "mae", "rel_err").

        Raises:
            ValueError: If an unknown metric is requested.

        Returns:
            Dictionary mapping metric names to computed float values.
        """
        if metrics is None:
            metrics = ["mse", "rmse", "rel_err"]
        if device is None:
            device = torch.device("cpu")
        y_pred = self.predict(raw_input, batch_size=batch_size, device=device)
        diff = y_pred - y_true
        epsilon = 1e-12
        results: dict[str, float] = {}
        for m in metrics:
            if m == "mse":
                results["mse"] = float(np.mean(diff**2))
            elif m == "rmse":
                results["rmse"] = float(np.sqrt(np.mean(diff**2)))
            elif m == "mae":
                results["mae"] = float(np.mean(np.abs(diff)))
            elif m == "rel_err":
                results["rel_err"] = float(np.mean(np.abs(diff) / (np.abs(y_true) + epsilon)))
            else:
                raise ValueError(f"Unknown metric {m}")
        return results

    def save(self, filepath: str, device: torch.device | None = None) -> None:
        """Saves the entire pipeline to a file.

        Moves models to the specified device, pickles the pipeline, and writes it to filepath.

        Args:
            filepath: Path where the pipeline is saved.
            device: Device to move models onto before saving. Defaults to CPU if None.

        Raises:
            OSError: If an error occurs during file write.
        """
        if device is None:
            device = torch.device("cpu")
        self.to_device(device)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> ModelPipeline:
        """Loads a serialized pipeline from a file.

        The pipeline includes trained PyTorch models, fitted scalers, and a DataPreprocessor.

        Args:
            filepath: Path to the pickle file containing the pipeline.

        Raises:
            FileNotFoundError: If the specified file does not exist.

        Returns:
            An instance of ModelPipeline.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)
