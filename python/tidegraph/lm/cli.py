"""Installed external-LM entry; help/catalog/fetch do not import Torch."""
import argparse
import json
from .catalog import catalog


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tidegraph lm", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog", help="show fixed model revisions and asset sizes")
    fetch = commands.add_parser("fetch", help="download/verify one model; rerun explicitly to resume partial files")
    fetch.add_argument("model", choices=tuple(catalog()))
    fetch.add_argument("--root", required=True, help="shared model asset directory")
    fetch.add_argument("--project-root", required=True, help="whole experiment tree included in disk accounting")
    fetch.add_argument("--project-gib", type=int, default=80)
    fetch.add_argument("--account-gib", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            result = {key: dict(id=value["id"], revision=value["revision"], stage=value["stage"],
                        bytes=sum(f["bytes"] for f in value["files"])) for key, value in catalog().items()}
        else:
            from .assets import fetch as run
            result = run(args.model, args.root, args.project_root,
                         project_gib=args.project_gib, account_gib=args.account_gib)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
