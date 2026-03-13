from typing import Dict, List, Optional, Tuple
import numpy as np
import sys
import os

PERSONALITY_HEADERS = [
    "Anxiety and Stress Levels",
    "Emotional Stability",
    "Problem-solving Skills",
    "Creativity",
    "Interpersonal Relationships",
    "Confidence and Self-efficacy",
    "Conflict Resolution",
    "Work-related Stress",
    "Adaptability",
    "Achievement Motivation",
    "Fear of Failure",
    "Need for Control",
    "Cognitive Load",
    "Social Support",
    "Resilience",
]
DEFAULT_EMBEDDING_METHOD = os.environ.get("PARSE_COMPUTE_METRICS_EMBEDDING", "pca").strip().lower()


def parse_markdown_table(file_path: str) -> Tuple[Dict[str, Dict[str, float]], List[float]]:
    result = {}
    std_values = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().split('\n')
        in_table = False
        header_found = False

        for line in lines:
            if line.strip().startswith('| LLM'):
                in_table = True
                header_found = True
                continue
            if in_table and is_markdown_separator_row(line):
                continue
            if in_table and line.strip().startswith('|'):
                columns = [col.strip() for col in line.split('|')[1:-1]]
                llm_name = columns[0].strip()
                metrics = {}
                for header, value in zip(PERSONALITY_HEADERS, columns[2:]):
                    this_mean = float(value.split("$")[0])
                    this_std = float(value.split("$")[2])
                    std_values.append(this_std)
                    metrics[header] = this_mean
                result[llm_name] = metrics
            if in_table and not line.strip().startswith('|') and header_found:
                break
        return result, std_values

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_path}' not found")
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")


def is_markdown_separator_row(line: str) -> bool:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False

    columns = [col.strip() for col in stripped.split("|")[1:-1]]
    if not columns:
        return False

    for column in columns:
        if not column:
            return False
        if any(char not in "-:" for char in column):
            return False

    return True


def build_metric_matrix(data: Dict[str, Dict[str, float]]) -> Tuple[List[str], np.ndarray]:
    labels = list(data.keys())
    matrix = np.asarray(
        [[data[label][header] for header in PERSONALITY_HEADERS] for label in labels],
        dtype=float,
    )
    return labels, matrix


def standardize_columns(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    safe_stds = np.where(stds == 0.0, 1.0, stds)
    return (matrix - means) / safe_stds


def compute_axis_std(matrix: np.ndarray, axis: int) -> np.ndarray:
    axis_length = matrix.shape[axis]
    if axis_length <= 1:
        output_size = matrix.shape[1 - axis]
        return np.zeros(output_size, dtype=float)
    return matrix.std(axis=axis, ddof=1)


def compute_pca_embedding(standardized_data: np.ndarray) -> np.ndarray:
    if len(standardized_data) == 0:
        return np.empty((0, 2), dtype=float)
    if len(standardized_data) == 1:
        return np.zeros((1, 2), dtype=float)

    _, _, vt = np.linalg.svd(standardized_data, full_matrices=False)
    components = vt[:2].T
    embedding = standardized_data @ components

    if embedding.shape[1] == 1:
        padding = np.zeros((embedding.shape[0], 1), dtype=float)
        embedding = np.hstack([embedding, padding])

    return embedding[:, :2]


def compute_embedding(standardized_data: np.ndarray) -> Tuple[str, np.ndarray]:
    if DEFAULT_EMBEDDING_METHOD == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42, metric='euclidean')
            return "umap", reducer.fit_transform(standardized_data)
        except Exception:
            pass

    return "pca", compute_pca_embedding(standardized_data)


def _format_tree_from_scipy(Z: np.ndarray, labels: List[str]) -> Tuple[str, int]:
    """
    Build a textual tree from a SciPy linkage matrix using '#' levels.
    Returns (text, max_depth).
    """
    from scipy.cluster.hierarchy import to_tree  # type: ignore

    root, _ = to_tree(Z, rd=True)

    # Recursively traverse the tree
    def rec(node, depth: int) -> Tuple[List[str], int]:
        if node.is_leaf():
            line = f"{'#' * depth} {labels[node.id]}"
            return [line], depth
        left_lines, left_max = rec(node.get_left(), depth + 1)
        right_lines, right_max = rec(node.get_right(), depth + 1)
        count = node.count
        height = float(node.dist) if hasattr(node, "dist") else float("nan")
        header = f"{'#' * depth} Cluster (n={count}, height={height:.3f})"
        lines = [] + left_lines + right_lines
        return lines, max(depth, left_max, right_max)

    lines, max_depth = rec(root, 1)
    return "\n".join(lines), max_depth


def _format_tree_from_children(
    children: np.ndarray,
    labels: List[str],
    distances: Optional[np.ndarray] = None
) -> Tuple[str, int]:
    """
    Build a textual hierarchy from scikit-learn AgglomerativeClustering 'children_' array.
    Returns (text, max_depth).
    """
    n_samples = len(labels)

    # Recursive traversal that returns (lines, count, max_depth)
    def rec(node_id: int, depth: int) -> Tuple[List[str], int, int]:
        if node_id < n_samples:
            # Leaf
            return [f"{'#' * depth} {labels[node_id]}"], 1, depth
        # Internal node index in children_ array
        row = node_id - n_samples
        left, right = children[row]
        left_lines, left_count, left_max = rec(left, depth + 1)
        right_lines, right_count, right_max = rec(right, depth + 1)
        count = left_count + right_count
        if distances is not None and len(distances) > row:
            header = f"{'#' * depth} Cluster (n={count}, height={distances[row]:.3f})"
        else:
            header = f"{'#' * depth} Cluster (n={count})"
        lines = [header] + left_lines + right_lines
        return lines, count, max(depth, left_max, right_max)

    root_id = n_samples + children.shape[0] - 1
    lines, _, max_depth = rec(root_id, 1)
    return "\n".join(lines), max_depth


def _build_hierarchical_text(
    standardized_data: np.ndarray,
    labels: List[str],
    method: str = "ward",
    metric: str = "euclidean"
) -> Dict[str, object]:
    """
    Compute hierarchical clustering and return a hash-prefixed textual representation.
    Tries SciPy first; falls back to scikit-learn if SciPy isn't available.
    """
    # Try SciPy first
    try:
        from scipy.cluster.hierarchy import linkage  # type: ignore
        Z = linkage(standardized_data, method=method, metric=metric)
        text_tree, max_depth = _format_tree_from_scipy(Z, labels)
        return {
            "method": method,
            "metric": metric,
            "max_depth": int(max_depth),
            "text_tree": text_tree
        }
    except Exception:
        pass  # Fall back to scikit-learn

    # Fallback: scikit-learn AgglomerativeClustering
    try:
        from sklearn.cluster import AgglomerativeClustering
        # Build the full tree
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.0,
            linkage=method if method in ("ward", "average", "complete", "single") else "ward",
        )
        model.fit(standardized_data)
        distances = getattr(model, "distances_", None)
        text_tree, max_depth = _format_tree_from_children(model.children_, labels, distances)
        return {
            "method": method if method in ("ward", "average", "complete", "single") else "ward",
            "metric": metric,
            "max_depth": int(max_depth),
            "text_tree": text_tree
        }
    except Exception:
        # Last-resort message if neither SciPy nor modern scikit-learn are available
        return {
            "method": method,
            "metric": metric,
            "max_depth": 0,
            "text_tree": "# Hierarchical clustering could not be computed (missing SciPy and modern scikit-learn)."
        }


def analyze_variability(data: Dict[str, Dict[str, float]]) -> Dict:
    """
    Compute various variability metrics and append a textual representation
    of hierarchical clustering between models.
    """
    labels, matrix = build_metric_matrix(data)
    standardized_data = standardize_columns(matrix)
    column_std = compute_axis_std(matrix, axis=0)

    # Initialize results dictionary
    results = {}

    # 1. Basic statistics per metric
    results['basic_stats'] = {
        'mean': dict(zip(PERSONALITY_HEADERS, matrix.mean(axis=0).tolist())),
        'std': dict(zip(PERSONALITY_HEADERS, column_std.tolist())),
        'coeff_variation': dict(
            zip(
                PERSONALITY_HEADERS,
                np.divide(
                    column_std,
                    matrix.mean(axis=0),
                    out=np.zeros(matrix.shape[1], dtype=float),
                    where=matrix.mean(axis=0) != 0,
                ).tolist(),
            )
        ),
        'range': dict(zip(PERSONALITY_HEADERS, (matrix.max(axis=0) - matrix.min(axis=0)).tolist()))
    }

    # 2. Embedding analysis
    embedding_method, embedding = compute_embedding(standardized_data)
    embedding_dict = {
        label: {"X1": float(coords[0]), "X2": float(coords[1])}
        for label, coords in zip(labels, embedding)
    }
    results['embedding'] = {
        'method': embedding_method,
        'embedding': embedding_dict,
        'variance_x1': float(np.var(embedding[:, 0])) if len(embedding) else 0.0,
        'variance_x2': float(np.var(embedding[:, 1])) if len(embedding) else 0.0,
        'total_variance': float(np.var(embedding[:, 0]) + np.var(embedding[:, 1])) if len(embedding) else 0.0,
        'range_x1': float(embedding[:, 0].max() - embedding[:, 0].min()) if len(embedding) else 0.0,
        'range_x2': float(embedding[:, 1].max() - embedding[:, 1].min()) if len(embedding) else 0.0,
    }

    # 3. Correlation analysis
    correlation_matrix = np.corrcoef(matrix, rowvar=False)
    if correlation_matrix.ndim == 0:
        correlation_matrix = np.asarray([[1.0]])
    abs_correlation = np.abs(correlation_matrix)
    mask = ~np.eye(abs_correlation.shape[0], dtype=bool)
    upper_i, upper_j = np.triu_indices_from(correlation_matrix, k=1)
    high_corr_mask = np.abs(correlation_matrix[upper_i, upper_j]) > 0.7
    results['correlation'] = {
        'mean_abs_correlation': float(abs_correlation[mask].mean()) if mask.any() else 0.0,
        'max_correlation': float(abs_correlation[mask].max()) if mask.any() else 0.0,
        'highly_correlated_pairs': [
            (
                PERSONALITY_HEADERS[i],
                PERSONALITY_HEADERS[j],
                float(correlation_matrix[i, j]),
            )
            for i, j in zip(upper_i[high_corr_mask], upper_j[high_corr_mask])
        ]
    }

    # 4. Variability across LLMs
    per_llm_var = matrix.var(axis=1, ddof=1) if matrix.shape[1] > 1 else np.zeros(len(labels), dtype=float)
    max_var_idx = int(np.argmax(per_llm_var)) if len(per_llm_var) else 0
    min_var_idx = int(np.argmin(per_llm_var)) if len(per_llm_var) else 0
    results['llm_variability'] = {
        'total_variance_per_llm': {
            label: float(value) for label, value in zip(labels, per_llm_var)
        },
        'mean_variance': float(per_llm_var.mean()) if len(per_llm_var) else 0.0,
        'max_variance_llm': str(labels[max_var_idx]) if labels else "",
        'min_variance_llm': str(labels[min_var_idx]) if labels else ""
    }

    # 5. Hierarchical clustering (textual tree appended in the output file)
    # Use the same standardized data to ensure scale-invariant clustering.
    hierarchical = _build_hierarchical_text(
        standardized_data=standardized_data,
        labels=labels,
        method="ward",
        metric="euclidean"
    )
    results['hierarchical'] = hierarchical

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

        embedding_name = variability_metrics['embedding']['method'].upper()
        f.write(f"\n2. {embedding_name} Analysis:\n")
        f.write(f"  Variance X1: {variability_metrics['embedding']['variance_x1']:.3f}\n")
        f.write(f"  Variance X2: {variability_metrics['embedding']['variance_x2']:.3f}\n")
        f.write(f"  Total variance: {variability_metrics['embedding']['total_variance']:.3f}\n")
        f.write(f"  Range X1: {variability_metrics['embedding']['range_x1']:.3f}\n")
        f.write(f"  Range X2: {variability_metrics['embedding']['range_x2']:.3f}\n")
        f.write("  Sample embeddings (first 3 LLMs):\n")
        for llm, coords in list(variability_metrics['embedding']['embedding'].items())[:3]:
            f.write(f"    {llm}: X1={coords['X1']:.3f}, X2={coords['X2']:.3f}\n")

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

        # 5. Append hierarchical clustering textual tree at the end
        f.write("\n5. Hierarchical Clustering (textual tree):\n")
        method = variability_metrics.get('hierarchical', {}).get('method', 'ward')
        metric = variability_metrics.get('hierarchical', {}).get('metric', 'euclidean')
        max_depth = variability_metrics.get('hierarchical', {}).get('max_depth', 0)
        f.write(f"  Method: {method}, Metric: {metric}, Max depth (levels): {max_depth}\n\n")
        text_tree = variability_metrics.get('hierarchical', {}).get('text_tree', '')
        # Ensure the tree ends with a newline
        f.write(text_tree.rstrip() + "\n")


def main(input_path: str, output_path: str):
    """
    Main function to analyze variability in LLM metrics.

    Args:
        input_path (str): Path to input markdown file
        output_path (str): Path to output file for results
    """
    try:
        # Parse the table
        parsed_data, std_values = parse_markdown_table(input_path)
        std_values.sort()

        # Calculate standard deviation statistics
        std_min = std_values[0]
        std_max = std_values[-1]
        std_median = std_values[int((len(std_values) - 1) * 0.5)]
        std_first = std_values[int((len(std_values) - 1) * 0.25)]
        std_third = std_values[int((len(std_values) - 1) * 0.75)]
        std_stats = (std_min, std_first, std_median, std_third, std_max)

        # Analyze variability (includes hierarchical clustering text)
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
        output_path = "stats/stats-" + judge.replace("/", "_") + ".txt"

        main(input_path, output_path)
