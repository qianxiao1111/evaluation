import json
from metrics import Metric
from utils import load_json
import argparse
import warnings

warnings.filterwarnings(action="ignore")


def evaluate_metrics(
    pred_tables_path, label_tables_path, pred_columns_path, label_columns_path
):
    pred_tables = load_json(pred_tables_path)
    pred_columns = load_json(pred_columns_path)
    label_tables = load_json(label_tables_path)
    label_columns = load_json(label_columns_path)

    def combine_metrics_under_key(pred_data, label_data, key):
        combined_metrics = {}
        for metric_name in ["averaged", "jaccard", "hamming"]:
            metric_results = getattr(Metric, metric_name)(pred_data, label_data)
            combined_metrics.update(metric_results)

        return {key: combined_metrics}

    table_metrics_combined = combine_metrics_under_key(
        pred_tables, label_tables, "table"
    )
    column_metrics_combined = combine_metrics_under_key(
        pred_columns, label_columns, "column"
    )
    merged_results = {**table_metrics_combined, **column_metrics_combined}
    return merged_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics.")
    parser.add_argument(
        "--pred_tables", required=True, help="Path to predicted tables JSON file."
    )
    parser.add_argument(
        "--label_tables", required=True, help="Path to label tables JSON file."
    )
    parser.add_argument(
        "--pred_columns", required=True, help="Path to predicted columns JSON file."
    )
    parser.add_argument(
        "--label_columns", required=True, help="Path to label columns JSON file."
    )

    args = parser.parse_args()

    results = evaluate_metrics(
        args.pred_tables, args.label_tables, args.pred_columns, args.label_columns
    )
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
