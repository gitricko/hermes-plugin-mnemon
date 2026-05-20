import sys
import json
import os
from pathlib import Path
from mnemon import _run_mnemon

def handle_mnemon_command(args, parser):
    cmd = getattr(args, "mnemon_cmd", None)
    if not cmd:
        parser.print_help()
        sys.exit(1)

    if cmd == "status":
        code, stdout, stderr = _run_mnemon(["--version"])
        if code != 0:
            print("Status: ERROR")
            print("mnemon binary is not available or not on PATH.")
            print(f"Error details: {stderr.strip()}")
            sys.exit(1)

        print("Status: ACTIVE")
        print(f"Version: {stdout.strip()}")

        active_store = os.environ.get("MNEMON_STORE")
        if not active_store:
            hermes_home = Path.home() / ".hermes"
            config_file = hermes_home / "mnemon.json"
            if config_file.exists():
                try:
                    config_data = json.loads(config_file.read_text())
                    active_store = config_data.get("store")
                except Exception:
                    pass

        if not active_store:
            active_store = "default (or derived from agent_identity)"

        print(f"Active Store: {active_store}")

        code, stdout, stderr = _run_mnemon(["store", "list"])
        if code == 0:
            print("\nAvailable stores:")
            print(stdout.strip())
        sys.exit(0)

    elif cmd == "config":
        hermes_home = Path.home() / ".hermes"
        config_file = hermes_home / "mnemon.json"
        print(f"Configuration File: {config_file}")
        if config_file.exists():
            print("Configuration:")
            try:
                print(json.dumps(json.loads(config_file.read_text()), indent=2))
            except Exception:
                print(config_file.read_text().strip())
        else:
            print("No custom configuration file found (using defaults).")
        sys.exit(0)

    elif cmd == "forget":
        insight_id = args.insight_id
        if not insight_id:
            print("Error: Please provide an insight_id.")
            sys.exit(1)

        code, stdout, stderr = _run_mnemon(["forget", insight_id])
        if code == 0:
            print(f"Successfully requested soft-delete for insight ID: {insight_id}")
            hermes_home = Path.home() / ".hermes"
            index_path = hermes_home / "mnemon_id_index.json"
            if index_path.exists():
                try:
                    idx = json.loads(index_path.read_text())
                    if insight_id in idx.get("ids", {}):
                        idx.get("ids", {}).pop(insight_id, None)
                        index_path.write_text(json.dumps(idx, indent=2))
                        print("Removed from local ID index.")
                except Exception as e:
                    print(f"Note: failed to clean up local index: {e}")
            sys.exit(0)
        else:
            print(f"Error: forget failed with code {code}")
            print(stderr.strip())
            sys.exit(1)

def register_cli(subparser):
    parser = subparser.add_parser("mnemon", help="Mnemon memory provider commands")
    subs = parser.add_subparsers(dest="mnemon_cmd")

    subs.add_parser("status", help="Show mnemon memory provider status and active store")
    subs.add_parser("config", help="Show mnemon memory provider configuration")

    forget_parser = subs.add_parser("forget", help="Soft-delete an insight by ID")
    forget_parser.add_argument("insight_id", help="The UUID of the insight to soft-delete")

    parser.set_defaults(func=lambda args: handle_mnemon_command(args, parser))
