from pydantic import BaseModel, model_validator
from typing import Dict
import json


class Prompt(BaseModel):
    prompt: str

    @model_validator(mode="after")
    def check_prompt(self):
        if not self.prompt.strip():
            raise ValueError("invalid")
        return self


class Parameter(BaseModel):
    type: str

    @model_validator(mode="after")
    def check_type(self):
        if not self.type.strip():
            raise ValueError("invalid")
        return self


class Returns(BaseModel):
    type: str

    @model_validator(mode="after")
    def check_type(self):
        if not self.type.strip():
            raise ValueError("invalid")
        return self


class Functions(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Returns

    @model_validator(mode="after")
    def check_function(self):
        if not self.name.strip():
            raise ValueError("invalid name")

        if not self.description.strip():
            raise ValueError("invalid description")

        return self


def main():
    with open(
        "/goinfre/cramadan/project/data/input/function_calling_tests.json", "r"
    ) as file:
        data = json.load(file)
    prompts = []
    i = 0
    while i < len(data):
        prompt = Prompt.model_validate(data[i])
        prompts.append(prompt)
        i += 1
    with open(
        "/goinfre/cramadan/project/data/input/functions_definition.json", "r"
    ) as file:
        data_function = json.load(file)
    functions = []
    i = 0
    while i < len(data_function):
        func = Functions.model_validate(data_function[i])
        functions.append(func)
        i += 1


def parsing_part():
    try:
        main()
    except Exception as e:
        print("Error:", e)


parsing_part()
