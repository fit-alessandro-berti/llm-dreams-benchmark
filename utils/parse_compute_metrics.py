from typing import Dict
import numpy as np
import pandas as pd
import sys
import os

ALL_STDS = []


def parse_markdown_table(file_path: str) -> Dict[str, Dict[str, float]]:
    """[Previous parsing function remains the same]"""
    result = {}
    headers = [
        "Anxiety and Stress Levels", "Emotional Stability", "Problem-solving Skills",
        "Creativity", "Interpersonal Relationships", "Confidence and Self-efficacy",
        "Conflict Resolution", "Work-related Stress", "Adaptability",
        "Achievement Motivation", "Fear of Failure", "Need for Control",
        "Cognitive Load", "Social Support", "Resilience"
    ]

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        lines = content.split('\n')
        in_table = False
        header_found = False

        for line in lines:
            if line.strip().startswith('| LLM'):
                in_table = True
                header_found = True
                continue
            if in_table and line.strip().startswith('|:--'):
                continue
            if in_table and line.strip().startswith('|'):
                columns = [col.strip() for col in line.split('|')[1:-1]]
                llm_name = columns[0].strip()
                metrics = {}
                for header, value in zip(headers, columns[2:]):
                    this_mean = float(value.split("$")[0])
                    this_std = float(value.split("$")[2])
                    ALL_STDS.append(this_std)
                    metrics[header] = this_mean
                result[llm_name] = metrics
            if in_table and not line.strip().startswith('|') and header_found:
                break
        ALL_STDS.sort()
        return result

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_path}' not found")
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")


def analyze_variability(data: Dict[str, Dict[str, float]]) -> Dict:
    """
    Compute various variability metrics for the parsed data using UMAP instead of PCA.

    Args:
        data (Dict[str, Dict[str, float]]): Parsed LLM metrics

    Returns:
        Dict: Dictionary containing various variability metrics
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data).T

    # Initialize results dictionary
    results = {}

    # 1. Basic statistics per metric
    results['basic_stats'] = {
        'mean': df.mean().to_dict(),
        'std': df.std().to_dict(),
        'coeff_variation': (df.std() / df.mean()).to_dict(),
        'range': (df.max() - df.min()).to_dict()
    }

    # 2. UMAP Analysis
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df)

    # Apply UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, metric='euclidean')
    embedding = reducer.fit_transform(standardized_data)

    # Compute variability metrics from UMAP embedding
    embedding_df = pd.DataFrame(embedding, index=df.index, columns=['UMAP1', 'UMAP2'])
    results['umap'] = {
        'embedding': embedding_df.to_dict(orient='index'),  # 2D coordinates for each LLM
        'variance_umap1': np.var(embedding[:, 0]),
        'variance_umap2': np.var(embedding[:, 1]),
        'total_variance': np.var(embedding[:, 0]) + np.var(embedding[:, 1]),
        'range_umap1': float(embedding[:, 0].max() - embedding[:, 0].min()),
        'range_umap2': float(embedding[:, 1].max() - embedding[:, 1].min())
    }

    # 3. Correlation analysis
    correlation_matrix = df.corr()
    results['correlation'] = {
        'mean_abs_correlation': correlation_matrix.abs().mean().mean(),
        'max_correlation': correlation_matrix.abs().where(~np.eye(len(df.columns), dtype=bool)).max().max(),
        'highly_correlated_pairs': [
            (col1, col2, correlation_matrix.loc[col1, col2])
            for col1 in correlation_matrix.columns
            for col2 in correlation_matrix.columns
            if col1 < col2 and abs(correlation_matrix.loc[col1, col2]) > 0.7
        ]
    }

    # 4. Variability across LLMs
    results['llm_variability'] = {
        'total_variance_per_llm': df.var(axis=1).to_dict(),
        'mean_variance': df.var(axis=1).mean(),
        'max_variance_llm': df.var(axis=1).idxmax(),
        'min_variance_llm': df.var(axis=1).idxmin()
    }

    return results


def write_results(output_path: str, variability_metrics: Dict, std_stats: tuple):
    """
    Write all results to the specified output file.

    Args:
        output_path (str): Path to the output file
        variability_metrics (Dict): Computed variability metrics
        std_stats (tuple): Standard deviation statistics (min, first_quart, median, third_quart, max)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write standard deviation statistics
        f.write("STD of single entries:\tmin=%.3f\t1st_quart=%.3f\tmedian=%.3f\t3rd_quart=%.3f\tmax=%.3f\n" % std_stats)

        # Write variability analysis results
        f.write("\nVariability Analysis Results (considering the AVG in each AVG ± STDEV tabular value):\n")
        f.write("\n1. Basic Statistics:\n")
        for stat, values in variability_metrics['basic_stats'].items():
            f.write(f"\n{stat.capitalize()}:\n")
            for metric, value in values.items():
                f.write(f"  {metric}: {value:.3f}\n")

        f.write("\n2. UMAP Analysis:\n")
        f.write(f"  Variance UMAP1: {variability_metrics['umap']['variance_umap1']:.3f}\n")
        f.write(f"  Variance UMAP2: {variability_metrics['umap']['variance_umap2']:.3f}\n")
        f.write(f"  Total variance: {variability_metrics['umap']['total_variance']:.3f}\n")
        f.write(f"  Range UMAP1: {variability_metrics['umap']['range_umap1']:.3f}\n")
        f.write(f"  Range UMAP2: {variability_metrics['umap']['range_umap2']:.3f}\n")
        f.write("  Sample embeddings (first 3 LLMs):\n")
        for llm, coords in list(variability_metrics['umap']['embedding'].items())[:3]:
            f.write(f"    {llm}: UMAP1={coords['UMAP1']:.3f}, UMAP2={coords['UMAP2']:.3f}\n")

        f.write("\n3. Correlation Analysis:\n")
        f.write(f"  Mean absolute correlation: {variability_metrics['correlation']['mean_abs_correlation']:.3f}\n")
        f.write(f"  Maximum correlation: {variability_metrics['correlation']['max_correlation']:.3f}\n")
        f.write("  Highly correlated pairs (|r| > 0.7):\n")
        for pair in variability_metrics['correlation']['highly_correlated_pairs']:
            f.write(f"    {pair[0]} - {pair[1]}: {pair[2]:.3f}\n")

        f.write("\n4. LLM Variability:\n")
        f.write(f"  Mean variance across LLMs: {variability_metrics['llm_variability']['mean_variance']:.3f}\n")
        f.write(f"  Max variance LLM: {variability_metrics['llm_variability']['max_variance_llm']}\n")
        f.write(f"  Min variance LLM: {variability_metrics['llm_variability']['min_variance_llm']}\n")


def main(input_path: str, output_path: str):
    """
    Main function to analyze variability in LLM metrics.

    Args:
        input_path (str): Path to input markdown file
        output_path (str): Path to output file for results
    """
    try:
        # Parse the table
        parsed_data = parse_markdown_table(input_path)

        # Calculate standard deviation statistics
        std_min = ALL_STDS[0]
        std_max = ALL_STDS[-1]
        std_median = ALL_STDS[int((len(ALL_STDS) - 1) * 0.5)]
        std_first = ALL_STDS[int((len(ALL_STDS) - 1) * 0.25)]
        std_third = ALL_STDS[int((len(ALL_STDS) - 1) * 0.75)]
        std_stats = (std_min, std_first, std_median, std_third, std_max)

        # Analyze variability
        variability_metrics = analyze_variability(parsed_data)

        # Write results to file
        write_results(output_path, variability_metrics, std_stats)

        print(f"Analysis complete. Results written to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    os.chdir("..")
    from common import ALL_JUDGES

    for index, judge in enumerate(ALL_JUDGES):
        print(index, judge)

        input_path = ALL_JUDGES[judge]["git_table_result"]
        output_path = "stats/stats-"+judge.replace("/", "_")+".txt"

        main(input_path, output_path)
