import argparse
from segger.prediction.predict import load_model, predict


def main(args: argparse.Namespace) -> None:
    """
    Main function to load the model and perform predictions.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    litsegger = load_model(
        args.checkpoint_path,
        args.init_emb,
        args.hidden_channels,
        args.out_channels,
        args.heads,
        args.aggr,
    )
    predict(
        litsegger,
        args.dataset_path,
        args.output_path,
        args.score_cut,
        args.k_nc,
        args.dist_nc,
        args.k_tx,
        args.dist_tx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using the Segger model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the predictions",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument("--init_emb", type=int, default=8, help="Initial embedding size")
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
        help="Number of hidden channels",
    )
    parser.add_argument("--out_channels", type=int, default=16, help="Number of output channels")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--aggr", type=str, default="sum", help="Aggregation method")
    parser.add_argument(
        "--score_cut",
        type=float,
        default=0.5,
        help="Score cut-off for predictions",
    )
    parser.add_argument(
        "--k_nc",
        type=int,
        default=4,
        help="Number of nearest neighbors for nuclei",
    )
    parser.add_argument("--dist_nc", type=int, default=20, help="Distance threshold for nuclei")
    parser.add_argument(
        "--k_tx",
        type=int,
        default=5,
        help="Number of nearest neighbors for transcripts",
    )
    parser.add_argument(
        "--dist_tx",
        type=int,
        default=10,
        help="Distance threshold for transcripts",
    )

    args = parser.parse_args()
    main(args)
