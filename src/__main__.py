"""CLI entry point: python -m grid_prophet <command>."""

import sys


def main():
    import argparse
    # Imported inside main() to avoid heavy import-time side effects at module load
    import collect
    import features
    import train
    import predict

    parser = argparse.ArgumentParser(
        prog="grid_prophet",
        description="Grid Prophet F1 championship predictor.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    sub.add_parser("collect", help="Pull historical race data from FastF1")
    sub.add_parser("features", help="Build feature matrix")
    sub.add_parser("train", help="Train model")
    sub.add_parser("predict", help="Generate 2026 predictions")
    sub.add_parser("run", help="Run full pipeline: collect > features > train > predict")
    sub.add_parser("update", help="Auto-detect latest round, rebuild features, retrain, and re-predict")
    sub.add_parser("plots", help="Generate all visualisation charts")

    args, remaining = parser.parse_known_args()
    prog_name = sys.argv[0]  # save before any mutation
    # Pass remaining args through to sub-module so their own argparse flags work
    sys.argv = [prog_name] + remaining

    if args.command == "collect":
        collect.main()
    elif args.command == "features":
        features.main()
    elif args.command == "train":
        train.main()
    elif args.command == "predict":
        predict.main()
    elif args.command == "run":
        from predict import PREDICT_YEAR
        collect.main()
        n = features.latest_completed_round(PREDICT_YEAR)
        sys.argv = [prog_name, "--early-rounds", str(n)]
        features.main()
        sys.argv = [prog_name]
        train.main()
        sys.argv = [prog_name, "--rounds", str(n)]
        predict.main()
    elif args.command == "update":
        from predict import PREDICT_YEAR
        n = features.latest_completed_round(PREDICT_YEAR)
        sys.argv = [prog_name, "--early-rounds", str(n)]
        features.main()
        sys.argv = [prog_name]
        train.main()
        sys.argv = [prog_name, "--rounds", str(n)]
        predict.main()
    elif args.command == "plots":
        try:
            import visualize
        except ImportError:
            print("Error: visualize module not yet available. Run all pipeline stages first.", file=sys.stderr)
            sys.exit(1)
        visualize.plot_all()


if __name__ == "__main__":
    main()
