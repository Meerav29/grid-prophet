"""CLI entry point: python -m grid_prophet <command>."""

import sys


def _detect_latest_round(predict_year: int) -> int:
    """Return the latest completed round number for predict_year from FastF1."""
    import fastf1
    import datetime
    schedule = fastf1.get_event_schedule(predict_year, include_testing=False)
    today = datetime.date.today()
    completed = schedule[schedule["EventDate"].dt.date < today]
    if completed.empty:
        return 1
    return int(completed["RoundNumber"].max())


def main():
    import argparse
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
    sub.add_parser("run", help="Run full pipeline: collect → features → train → predict")
    sub.add_parser("update", help="Auto-detect latest round and re-predict")
    sub.add_parser("plots", help="Generate all visualisation charts")

    args, remaining = parser.parse_known_args()
    # Pass remaining args through to sub-module so their own argparse flags work
    sys.argv = [sys.argv[0]] + remaining

    if args.command == "collect":
        collect.main()
    elif args.command == "features":
        features.main()
    elif args.command == "train":
        train.main()
    elif args.command == "predict":
        predict.main()
    elif args.command == "run":
        collect.main()
        features.main()
        train.main()
        predict.main()
    elif args.command == "update":
        from predict import PREDICT_YEAR
        n = _detect_latest_round(PREDICT_YEAR)
        sys.argv = [sys.argv[0], "--rounds", str(n)]
        predict.main()
    elif args.command == "plots":
        import visualize
        visualize.plot_all()


if __name__ == "__main__":
    main()
