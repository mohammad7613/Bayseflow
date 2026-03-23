"""
Utility functions for applying BayesFlow DDM models to real pedestrian crossing data.

This module provides helper functions for:
- Loading and preprocessing real experimental data
- Preparing data for BayesFlow inference
- Running inference on individual subjects
- Analyzing and visualizing results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os


def load_and_validate_data(
    filepath: str,
    required_columns: List[str] = ['subject_id', 'TTA', 'RT']
) -> pd.DataFrame:
    """
    Load experimental data and validate required columns exist.
    
    Args:
        filepath: Path to CSV file containing experimental data
        required_columns: List of column names that must be present
    
    Returns:
        DataFrame with validated data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✓ Loaded data: {len(df)} trials, {df['subject_id'].nunique()} subjects")
    print(f"  Columns: {list(df.columns)}")
    print(f"  TTA conditions: {sorted(df['TTA'].unique())}")
    
    return df


def prepare_subject_data(
    subject_df: pd.DataFrame,
    tta_column: str = 'TTA',
    rt_column: str = 'RT',
    cpp_column: Optional[str] = 'CPP',
    expected_ttas: List[float] = [2.5, 3.0, 3.5, 4.0]
) -> Dict[float, np.ndarray]:
    """
    Prepare one subject's data for BayesFlow inference.
    
    Args:
        subject_df: DataFrame containing one subject's trials
        tta_column: Name of column containing TTA values
        rt_column: Name of column containing reaction times
        cpp_column: Name of column containing CPP values (None if not available)
        expected_ttas: List of expected TTA values
    
    Returns:
        Dictionary mapping TTA -> array of shape (n_trials, 2) with [RT, CPP]
    """
    data_by_condition = {}
    
    for tta in expected_ttas:
        # Get trials for this TTA condition
        condition_mask = np.isclose(subject_df[tta_column], tta, atol=0.01)
        condition_trials = subject_df[condition_mask]
        
        if len(condition_trials) == 0:
            print(f"  ⚠ Warning: No trials found for TTA={tta}s")
            continue
        
        # Extract RT
        rts = condition_trials[rt_column].values
        
        # Extract or create CPP
        if cpp_column is not None and cpp_column in condition_trials.columns:
            cpps = condition_trials[cpp_column].values
        else:
            # Use zeros if CPP not available
            cpps = np.zeros_like(rts)
        
        # Validate data
        if np.any(np.isnan(rts)) or np.any(np.isnan(cpps)):
            n_nan = np.sum(np.isnan(rts)) + np.sum(np.isnan(cpps))
            print(f"  ⚠ Warning: {n_nan} NaN values in TTA={tta}s data")
            # Remove NaN trials
            valid_mask = ~(np.isnan(rts) | np.isnan(cpps))
            rts = rts[valid_mask]
            cpps = cpps[valid_mask]
        
        # Stack into [n_trials, 2] format
        x = np.column_stack([rts, cpps])
        data_by_condition[tta] = x
        
        print(f"  TTA={tta}s: {len(x)} trials prepared")
    
    return data_by_condition


def infer_subject_parameters(
    approximator,
    adapter,
    data_by_condition: Dict[float, np.ndarray],
    num_samples: int = 2000,
    combine_method: str = 'concatenate'
) -> Dict[str, np.ndarray]:
    """
    Infer subject-level parameters using data from all TTA conditions.
    
    Args:
        approximator: Trained BayesFlow approximator
        adapter: BayesFlow adapter function
        data_by_condition: Dict mapping TTA -> trial data array
        num_samples: Number of posterior samples per condition
        combine_method: How to combine posteriors ('concatenate' or 'average')
    
    Returns:
        Dictionary with posterior samples for each parameter
    """
    if not data_by_condition:
        raise ValueError("No data provided for inference")
    
    all_posteriors = []
    
    for tta, trials in data_by_condition.items():
        # Prepare data in BayesFlow format
        data_dict = {
            'x': trials[np.newaxis, :, :],  # Add batch dimension: [1, n_trials, 2]
            'tta_condition': np.array([tta]),
            'number_of_trials': np.array([len(trials)])
        }
        
        # Apply adapter transformations
        try:
            adapted_data = adapter(data_dict)
        except Exception as e:
            print(f"  ✗ Error adapting data for TTA={tta}s: {e}")
            continue
        
        # Get posterior samples
        try:
            posterior = approximator.sample(
                conditions=adapted_data,
                num_samples=num_samples
            )
            all_posteriors.append(posterior)
            print(f"  ✓ TTA={tta}s: {num_samples} posterior samples drawn")
        except Exception as e:
            print(f"  ✗ Error sampling posterior for TTA={tta}s: {e}")
            continue
    
    if not all_posteriors:
        raise RuntimeError("Failed to get posteriors for any condition")
    
    # Combine posteriors from all conditions
    combined_posterior = {}
    param_names = list(all_posteriors[0].keys())
    
    for param in param_names:
        if combine_method == 'concatenate':
            # Concatenate all samples (treats conditions as independent)
            combined_posterior[param] = np.concatenate(
                [p[param] for p in all_posteriors], axis=0
            )
        elif combine_method == 'average':
            # Average posterior means (assumes same parameters across conditions)
            stacked = np.stack([p[param].mean(axis=0) for p in all_posteriors])
            combined_posterior[param] = stacked
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")
    
    return combined_posterior


def summarize_posterior(
    posterior: Dict[str, np.ndarray],
    parameter_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute summary statistics for posterior distributions.
    
    Args:
        posterior: Dictionary of posterior samples
        parameter_names: List of parameters to include (None = all)
    
    Returns:
        DataFrame with summary statistics (mean, median, std, CI)
    """
    if parameter_names is None:
        parameter_names = list(posterior.keys())
    
    summaries = []
    
    for param in parameter_names:
        if param not in posterior:
            continue
        
        samples = posterior[param].flatten()
        
        summary = {
            'parameter': param,
            'mean': np.mean(samples),
            'median': np.median(samples),
            'std': np.std(samples),
            'ci_lower': np.percentile(samples, 2.5),
            'ci_upper': np.percentile(samples, 97.5),
            'ci_width': np.percentile(samples, 97.5) - np.percentile(samples, 2.5)
        }
        summaries.append(summary)
    
    return pd.DataFrame(summaries)


def process_all_subjects(
    df: pd.DataFrame,
    approximator,
    adapter,
    subject_col: str = 'subject_id',
    num_samples: int = 2000,
    save_results: bool = True,
    output_path: str = 'inferred_parameters_all_subjects.csv'
) -> pd.DataFrame:
    """
    Process all subjects in a dataset and infer parameters for each.
    
    Args:
        df: DataFrame containing data for all subjects
        approximator: Trained BayesFlow approximator
        adapter: BayesFlow adapter function
        subject_col: Name of column containing subject IDs
        num_samples: Number of posterior samples per subject
        save_results: Whether to save results to file
        output_path: Path to save results CSV
    
    Returns:
        DataFrame with parameter estimates for all subjects
    """
    unique_subjects = df[subject_col].unique()
    print(f"\nProcessing {len(unique_subjects)} subjects...\n")
    
    all_results = []
    
    for idx, subject_id in enumerate(unique_subjects, 1):
        print(f"[{idx}/{len(unique_subjects)}] Processing subject {subject_id}...")
        
        try:
            # Get subject data
            subject_data = df[df[subject_col] == subject_id]
            
            # Prepare data by condition
            data_by_condition = prepare_subject_data(subject_data)
            
            if not data_by_condition:
                print(f"  ✗ No valid data for subject {subject_id}, skipping\n")
                continue
            
            # Run inference
            posterior = infer_subject_parameters(
                approximator=approximator,
                adapter=adapter,
                data_by_condition=data_by_condition,
                num_samples=num_samples
            )
            
            # Summarize results
            summary = summarize_posterior(posterior)
            
            # Add subject ID to each row
            result = {'subject_id': subject_id}
            for _, row in summary.iterrows():
                param = row['parameter']
                result[f"{param}_mean"] = row['mean']
                result[f"{param}_median"] = row['median']
                result[f"{param}_std"] = row['std']
                result[f"{param}_ci_lower"] = row['ci_lower']
                result[f"{param}_ci_upper"] = row['ci_upper']
            
            all_results.append(result)
            print(f"  ✓ Subject {subject_id} completed\n")
            
        except Exception as e:
            print(f"  ✗ Error processing subject {subject_id}: {e}\n")
            continue
    
    if not all_results:
        raise RuntimeError("Failed to process any subjects")
    
    # Compile results
    results_df = pd.DataFrame(all_results)
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {len(results_df)} subjects successfully analyzed")
    print(f"{'='*60}\n")
    
    if save_results:
        results_df.to_csv(output_path, index=False)
        print(f"✓ Results saved to: {output_path}")
    
    return results_df


def plot_subject_posteriors(
    posterior: Dict[str, np.ndarray],
    parameter_names: List[str],
    true_values: Optional[Dict[str, float]] = None,
    title: str = "Posterior Distributions",
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Plot posterior distributions for a single subject.
    
    Args:
        posterior: Dictionary of posterior samples
        parameter_names: List of parameters to plot
        true_values: Optional dict of true parameter values (for simulated data)
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n_params = len(parameter_names)
    n_cols = 4
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, param in enumerate(parameter_names):
        if param not in posterior:
            continue
        
        ax = axes[idx]
        samples = posterior[param].flatten()
        
        # Plot histogram
        ax.hist(samples, bins=50, density=True, alpha=0.6, 
                color='blue', edgecolor='black')
        
        # Plot posterior mean
        mean_val = np.mean(samples)
        ax.axvline(mean_val, color='green', linestyle='-', 
                   linewidth=2, label=f'Mean: {mean_val:.3f}')
        
        # Plot true value if available
        if true_values is not None and param in true_values:
            ax.axvline(true_values[param], color='red', linestyle='--', 
                       linewidth=2, label=f'True: {true_values[param]:.3f}')
        
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.set_title(f"{param}")
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()
    
    return fig


def plot_group_comparison(
    results_df: pd.DataFrame,
    parameter: str,
    group_column: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot parameter estimates across subjects, optionally grouped.
    
    Args:
        results_df: DataFrame from process_all_subjects()
        parameter: Parameter name to plot (e.g., 'theta')
        group_column: Optional column name for grouping subjects
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    mean_col = f"{parameter}_mean"
    ci_lower_col = f"{parameter}_ci_lower"
    ci_upper_col = f"{parameter}_ci_upper"
    
    if mean_col not in results_df.columns:
        raise ValueError(f"Parameter '{parameter}' not found in results")
    
    x = np.arange(len(results_df))
    means = results_df[mean_col].values
    ci_lower = results_df[ci_lower_col].values
    ci_upper = results_df[ci_upper_col].values
    
    # Plot with error bars
    if group_column is not None and group_column in results_df.columns:
        groups = results_df[group_column].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        for group, color in zip(groups, colors):
            mask = results_df[group_column] == group
            ax.errorbar(
                x[mask], means[mask],
                yerr=[means[mask] - ci_lower[mask], ci_upper[mask] - means[mask]],
                fmt='o', capsize=5, label=f'{group_column}={group}',
                color=color
            )
    else:
        ax.errorbar(
            x, means,
            yerr=[means - ci_lower, ci_upper - means],
            fmt='o', capsize=5, color='blue'
        )
    
    ax.set_xlabel('Subject Index')
    ax.set_ylabel(f'{parameter} (posterior mean)')
    ax.set_title(f'{parameter} Estimates Across Subjects')
    ax.grid(True, alpha=0.3)
    
    if group_column is not None:
        ax.legend()
    
    plt.tight_layout()
    return fig


def export_for_statistical_analysis(
    results_df: pd.DataFrame,
    output_path: str = 'parameters_for_stats.csv',
    long_format: bool = True
) -> pd.DataFrame:
    """
    Export parameter estimates in format suitable for statistical analysis.
    
    Args:
        results_df: DataFrame from process_all_subjects()
        output_path: Path to save CSV
        long_format: If True, reshape to long format (one row per parameter per subject)
    
    Returns:
        DataFrame in requested format
    """
    if not long_format:
        results_df.to_csv(output_path, index=False)
        print(f"✓ Exported wide format to: {output_path}")
        return results_df
    
    # Reshape to long format
    param_cols = [col for col in results_df.columns if '_mean' in col]
    parameters = [col.replace('_mean', '') for col in param_cols]
    
    long_data = []
    
    for _, row in results_df.iterrows():
        subject_id = row['subject_id']
        
        for param in parameters:
            long_data.append({
                'subject_id': subject_id,
                'parameter': param,
                'mean': row[f'{param}_mean'],
                'median': row[f'{param}_median'],
                'std': row[f'{param}_std'],
                'ci_lower': row[f'{param}_ci_lower'],
                'ci_upper': row[f'{param}_ci_upper']
            })
    
    long_df = pd.DataFrame(long_data)
    long_df.to_csv(output_path, index=False)
    
    print(f"✓ Exported long format to: {output_path}")
    return long_df


# Example usage
if __name__ == "__main__":
    print("""
    Utility functions for BayesFlow DDM real data analysis.
    
    Example workflow:
    
    ```python
    import keras
    from bayesflow_models.DDM_DC_Pedestrain import all_models
    from bayesflow_models.utils_real_data import *
    
    # 1. Load data
    df = load_and_validate_data('pedestrian_data.csv')
    
    # 2. Load trained model
    model, adapter = all_models['model_DC']
    approximator = keras.saving.load_model('trained_model1/checkpoints/model_DC.keras')
    
    # 3. Process all subjects
    results = process_all_subjects(
        df=df,
        approximator=approximator,
        adapter=adapter,
        num_samples=2000
    )
    
    # 4. Analyze results
    print(results.describe())
    
    # 5. Export for stats
    export_for_statistical_analysis(results, long_format=True)
    ```
    """)
