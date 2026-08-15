from typing import Any, cast
from pathlib import Path
import argparse
import json
import sys

try:
    from llm_sdk.llm_sdk import Small_LLM_Model
except KeyboardInterrupt:
    sys.exit(0)


def read_vocab(llm: Small_LLM_Model) -> dict[str, int]:
    path = llm.get_path_to_vocab_file()
    try:
        with open(path, "r") as file:
            vocab = json.load(file)
    except FileNotFoundError as e:
        print("Error:", e)
        sys.exit(0)
    return cast(dict[str, int], vocab)


def write_output(output_text: list[Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output_text, file, indent=4)

    except Exception as error:
        print(f"Error writing output: {error}")


def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )

    args = parser.parse_args()
    return args
