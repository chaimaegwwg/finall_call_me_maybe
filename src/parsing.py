from pydantic import BaseModel, model_validator
from typing import Dict
import argparse
import json
import sys
import re


class Prompt(BaseModel):
    prompt: str

    @model_validator(mode="after")
    def check_prompt(self) -> "Prompt":
        if not self.prompt.strip():
            raise ValueError("invalid")
        return self


class Parameter(BaseModel):
    type: str

    @model_validator(mode="after")
    def check_type(self) -> "Parameter":
        if not self.type.strip():
            raise ValueError("invalid")
        return self


class Returns(BaseModel):
    type: str

    @model_validator(mode="after")
    def check_type(self) -> "Returns":
        if not self.type.strip():
            raise ValueError("invalid")
        return self


class Functions(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Returns

    @model_validator(mode="after")
    def check_function(self) -> "Functions":
        if not self.name.strip():
            raise ValueError("invalid name")

        if not self.description.strip():
            raise ValueError("invalid description")

        return self


def main(arg: argparse.Namespace) -> None:
    with open(
        arg.input, "r"
    ) as file:
        data = json.load(file)
    prompts = []
    i = 0
    while i < len(data):
        prompt = Prompt.model_validate(data[i])
        prompts.append(prompt)
        i += 1
    INT_max = 2147483647
    INT_min = -2147483648
    for p in prompts:
        num = re.findall(r"-?\d+", p.prompt)
        for k in num:
            if int(k) > INT_max or int(k) < INT_min:
                print(f"Error: the number {k}")
                sys.exit(0)
    with open(
        arg.functions_definition, "r"
    ) as file:
        data_function = json.load(file)
    functions = []
    i = 0
    while i < len(data_function):
        func = Functions.model_validate(data_function[i])
        functions.append(func)
        i += 1


def parsing_part(arg: argparse.Namespace) -> None:
    try:
        main(arg)
    except Exception as e:
        print("Error:", e)
        sys.exit(0)
