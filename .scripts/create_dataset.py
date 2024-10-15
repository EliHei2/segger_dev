import argparse
import os
from pathlib import Path
from urllib import request
from segger.data.io import XeniumSample


def download_file(url, dest):
    if not dest.exists():
        print(f"Downloading {url} to {dest}...")
        request.urlretrieve(url, dest)
        print("Download completed.")


def main(args):
    os.environ["USE_PYGEOS"] = "0"

    raw_data_dir = Path(args.raw_data_dir)
    processed_data_dir = Path(args.processed_data_dir)

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    transcripts_url = args.transcripts_url
    nuclei_url = args.nuclei_url

    transcripts_path = raw_data_dir / "transcripts.csv.gz"
    nuclei_path = raw_data_dir / "nucleus_boundaries.csv.gz"

    download_file(transcripts_url, transcripts_path)
    download_file(nuclei_url, nuclei_path)

    xs = XeniumSample().load_transcripts(path=transcripts_path, min_qv=args.min_qv)
    xs.load_nuclei(path=nuclei_path)

    if args.parallel:
        xs.save_dataset_for_segger_parallel(
            processed_data_dir,
            d_x=args.d_x,
            d_y=args.d_y,
            x_size=args.x_size,
            y_size=args.y_size,
            margin_x=args.margin_x,
            margin_y=args.margin_y,
            r_tx=args.r_tx,
            val_prob=args.val_prob,
            test_prob=args.test_prob,
            compute_labels=args.compute_labels,
            sampling_rate=args.sampling_rate,
            num_workers=args.num_workers,
            receptive_field={
                "k_nc": args.k_nc,
                "dist_nc": args.dist_nc,
                "k_tx": args.k_tx,
                "dist_tx": args.dist_tx,
            },
        )
    else:
        xs.save_dataset_for_segger(
            processed_data_dir,
            d_x=args.d_x,
            d_y=args.d_y,
            x_size=args.x_size,
            y_size=args.y_size,
            margin_x=args.margin_x,
            margin_y=args.margin_y,
            r_tx=args.r_tx,
            val_prob=args.val_prob,
            test_prob=args.test_prob,
            compute_labels=args.compute_labels,
            sampling_rate=args.sampling_rate,
            receptive_field={
                "k_nc": args.k_nc,
                "dist_nc": args.dist_nc,
                "k_tx": args.k_tx,
                "dist_tx": args.dist_tx,
            },
        )

    print("Dataset creation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset from Xenium Human Pancreatic data.")
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        help="Directory to store raw data.",
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        required=True,
        help="Directory to store processed data.",
    )
    parser.add_argument(
        "--transcripts_url",
        type=str,
        required=True,
        help="URL for transcripts data.",
    )
    parser.add_argument("--nuclei_url", type=str, required=True, help="URL for nuclei data.")
    parser.add_argument(
        "--min_qv",
        type=int,
        default=30,
        help="Minimum quality value for filtering transcripts.",
    )
    parser.add_argument(
        "--d_x",
        type=int,
        default=180,
        help="Step size in x direction for tiles.",
    )
    parser.add_argument(
        "--d_y",
        type=int,
        default=180,
        help="Step size in y direction for tiles.",
    )
    parser.add_argument("--x_size", type=int, default=200, help="Width of each tile.")
    parser.add_argument("--y_size", type=int, default=200, help="Height of each tile.")
    parser.add_argument("--margin_x", type=int, default=None, help="Margin in x direction.")
    parser.add_argument("--margin_y", type=int, default=None, help="Margin in y direction.")
    parser.add_argument("--r_tx", type=int, default=3, help="Radius for building the graph.")
    parser.add_argument(
        "--val_prob",
        type=float,
        default=0.1,
        help="Probability of assigning a tile to the validation set.",
    )
    parser.add_argument(
        "--test_prob",
        type=float,
        default=0.1,
        help="Probability of assigning a tile to the test set.",
    )
    parser.add_argument(
        "--k_nc",
        type=int,
        default=3,
        help="Number of nearest neighbors for nuclei.",
    )
    parser.add_argument("--dist_nc", type=int, default=10, help="Distance threshold for nuclei.")
    parser.add_argument(
        "--k_tx",
        type=int,
        default=5,
        help="Number of nearest neighbors for transcripts.",
    )
    parser.add_argument(
        "--dist_tx",
        type=int,
        default=3,
        help="Distance threshold for transcripts.",
    )
    parser.add_argument(
        "--compute_labels",
        type=bool,
        default=True,
        help="Whether to compute edge labels.",
    )
    parser.add_argument("--sampling_rate", type=float, default=1, help="Rate of sampling tiles.")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for parallel processing.",
    )

    args = parser.parse_args()
    main(args)
