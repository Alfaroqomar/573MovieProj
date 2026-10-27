"""Command-line entry points.

    python -m moviematch.cli status       # report readiness
    python -m moviematch.cli preprocess   # MovieLens CSVs -> processed artifacts
    python -m moviematch.cli train        # train popularity, item-KNN, ALS
    python -m moviematch.cli evaluate     # offline metrics on the test split
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from moviematch.config import get_settings


def _cmd_status(_: argparse.Namespace) -> int:
    from moviematch.service import readiness

    print(json.dumps(readiness(get_settings()).as_dict(), indent=2))
    return 0


def _cmd_preprocess(_: argparse.Namespace) -> int:
    from moviematch.data.preprocess import DatasetError, preprocess

    try:
        preprocess(get_settings())
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from moviematch.models.training import NotPreprocessedError, train_models

    try:
        train_models(get_settings(), only=args.only)
    except NotPreprocessedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from moviematch.models.evaluation import evaluate

    try:
        evaluate(get_settings(), split_name=args.split, k=args.k, sample_users=args.users)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="moviematch")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Report dataset, model, and LLM readiness").set_defaults(
        func=_cmd_status
    )
    sub.add_parser("preprocess", help="Validate and preprocess MovieLens 32M").set_defaults(
        func=_cmd_preprocess
    )

    train = sub.add_parser("train", help="Train and save recommendation models")
    train.add_argument(
        "--only", nargs="+", choices=["popularity", "knn", "als"], help="Train only these models"
    )
    train.set_defaults(func=_cmd_train)

    evaluate = sub.add_parser("evaluate", help="Compute offline ranking metrics")
    evaluate.add_argument("--split", choices=["test", "validation"], default="test")
    evaluate.add_argument("--k", type=int, default=10)
    evaluate.add_argument("--users", type=int, default=None, help="Number of users to sample")
    evaluate.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
