"""The single command registry for preparation, inference, training and experiments.

Importing this registry does not load models or start work. Each command forwards
its remaining arguments to the owner of that workflow.
"""

import argparse
import importlib
import sys

COMMANDS = {'prepare pages': ['data.screenshots', 'main'],
 'prepare grounding': ['data.grounding_inputs', 'main'],
 'infer proposals': ['retrieval.deberta', 'main'],
 'evaluate proposals': ['evaluation.proposals', 'main'],
 'infer grounding': ['grounding.inference', 'main'],
 'infer coordinate-evidence': ['data.evidence_collection', 'coordinate_evidence_main'],
 'experiment element-training': ['experiments.element_training', 'main'],
 'experiment element-controls': ['experiments.element_controls', 'controls_main'],
 'experiment element-features': ['experiments.element_controls', 'features_main'],
 'experiment element-seeds': ['experiments.element_controls', 'seeds_main'],
 'experiment element-mechanisms': ['experiments.element_mechanisms', 'mechanisms_main'],
 'experiment element-functional': ['experiments.element_mechanisms', 'functional_main'],
 'verify element-controls': ['evaluation.element_verification', 'controls_main'],
 'verify element-functional': ['evaluation.element_verification', 'functional_main'],
 'inspect inputs': ['manifest', 'main']}




def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="grounding",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + "\n".join("  " + key for key in sorted(COMMANDS)),
    )
    if not argv or argv == ["--help"] or argv == ["-h"]:
        parser.print_help()
        return 0
    command = " ".join(argv[:2])
    if command not in COMMANDS:
        parser.error(f"unknown command {command!r}; use --help to list commands")
    module, function = COMMANDS[command]
    original = sys.argv
    try:
        sys.argv = ["grounding " + command, *argv[2:]]
        return getattr(importlib.import_module(module), function)()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
