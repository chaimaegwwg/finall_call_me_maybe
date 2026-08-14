from llm_sdk.llm_sdk import Small_LLM_Model
import torch
import json
import argparse
import sys
from typing import Any
from pathlib import Path


class LLM:
    def __init__(self, llm: Small_LLM_Model, vocab: dict[str, int]) -> None:
        with open(
            '/goinfre/cramadan/project/data/input/functions_definition.json',
            'r'
        ) as file:
            self.functions = json.load(file)
        self.llm = llm
        self.vocab = vocab
        self.fixed_tokens = {
            "name": llm.encode("name").tolist()[0],
            "parameters": llm.encode("parameters").tolist()[0],
            "prompt": llm.encode("prompt").tolist()[0]
        }

    def all_functions(self) -> list[str]:
        lst = []

        for function in self.functions:
            lst.append(function["name"])
        # lst.append("unknown")
        return lst

    def parameter_type_func(
        self,
        function_name: str,
        parameter: str
    ) -> str | None:
        parameters = self.get_parameters(function_name)

        if parameters is None:
            return None

        if parameter not in parameters:
            return None

        return parameters[parameter]["type"]

    def get_parameters(
        self,
        function_name: str
    ) -> dict[str, Any] | None:
        for function in self.functions:
            if function["name"] == function_name:
                return function["parameters"]

        return None

    def ft_constrain_one_token(
        self,
        parameter: str,
        inputs: list[int],
        new_token: list[int]
    ) -> tuple[list[int], list[int]]:
        logits = self.llm.get_logits_from_input_ids(inputs)
        logits = torch.tensor(logits)
        wanted = self.llm.encode(parameter).tolist()[0][0]
        original_logits = logits.clone()
        logits[:] = float("-inf")
        logits[wanted] = original_logits[wanted]
        predicted_tensor = torch.argmax(logits)
        new_token.append(predicted_tensor.item())
        inputs.append(predicted_tensor.item())
        return new_token, inputs

    def ft_constrain(
        self,
        parameter: str,
        inputs: list[int],
        new_token: list[int]
    ) -> tuple[list[int], list[int]]:
        token = self.vocab[parameter]
        # print("here the debug",token)
        new_token.append(token)
        inputs.append(token)
        return new_token, inputs

    def ft_constrain_tokens(
        self,
        parameter: list[int],
        inputs: list[int],
        new_token: list[int]
    ) -> tuple[list[int], list[int]]:
        ids = parameter

        for token_id in ids:
            new_token.append(token_id)
            inputs.append(token_id)

        return new_token, inputs

    def ft_constrain_name_function(
        self,
        inputs: list[int],
        new_token: list[int],
        llm: Small_LLM_Model
    ) -> tuple[list[int], list[int], list[int]]:
        name_of_func = []
        functions = self.all_functions()
        lst_gath_func = []
        for function in functions:
            lst_gath_func.append(llm.encode(function).tolist()[0])
        while True:
            remove_lst = []
            lst_index = []
            if all(len(x) == 0 for x in lst_gath_func):
                break
            # if len(lst_gath_func) == 1 and len(lst_gath_func[0]) == 0:
            #     break
            for func in lst_gath_func:
                if len(func) <= 0:
                    remove_lst.append(func)
                    continue
                lst_index.append(func[0])

            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)
            original_logits = logits.clone()
            logits[:] = float("-inf")
            logits[lst_index] = original_logits[lst_index]
            predicted_tensor = torch.argmax(logits)
            for fun in lst_gath_func:
                predicted = predicted_tensor.item()
                if len(fun) == 0 or 0 >= len(fun) or fun[0] != predicted:
                    remove_lst.append(fun)
                else:
                    fun.pop(0)

            for fun in remove_lst:
                if fun not in lst_gath_func:
                    continue
                lst_gath_func.remove(fun)
            name_of_func.append(predicted_tensor.item())
            new_token.append(predicted_tensor.item())
            inputs.append(predicted_tensor.item())
        return new_token, inputs, name_of_func

    def ft_constrain_parameters(
        self,
        inputs: list[int],
        new_token: list[int],
        name_of_func: list[int],
        llm: Small_LLM_Model
    ) -> tuple[list[int], list[int], list[int]] | None:
        name = llm.decode(name_of_func).strip()
        parameters = self.get_parameters(name)
        if parameters is None:
            return
        name_of_parameter = []
        ids_lst = []
        for parameter in parameters:
            ids_lst.append(llm.encode(parameter).tolist()[0])

        while True:
            remove_lst = []
            lst_index = []
            if all(len(x) == 0 for x in ids_lst):
                break
            # if len(ids_lst) == 1 and len(ids_lst[0]) == 0:
            #     break
            for func in ids_lst:
                if len(func) <= 0:
                    remove_lst.append(func)
                    continue
                lst_index.append(func[0])

            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)
            original_logits = logits.clone()
            logits[:] = float("-inf")
            logits[lst_index] = original_logits[lst_index]
            predicted_tensor = torch.argmax(logits)

            for fun in ids_lst:
                predicted = predicted_tensor.item()
                if len(fun) == 0 or 0 >= len(fun) or fun[0] != predicted:
                    remove_lst.append(fun)
                else:
                    fun.pop(0)
            for fun in remove_lst:
                if fun not in ids_lst:
                    continue
                ids_lst.remove(fun)
            name_of_parameter.append(predicted_tensor.item())
            new_token.append(predicted_tensor.item())
            inputs.append(predicted_tensor.item())
        return new_token, inputs, name_of_parameter

    def ft_numb_num(
        self,
        inputs: list[int],
        new_token: list[int],
        llm: Small_LLM_Model,
        parameter_type: str
    ) -> tuple[list[int], list[int]]:
        number_tokens = []

        for _ in range(10):
            logits = llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)

            predicted_tensor = torch.argmax(logits)
            token_id = predicted_tensor.item()
            token_text = llm.decode([token_id])

            print("NUMBER TOKEN:", repr(token_text))

            if "," in token_text or "}" in token_text:
                break

            number_tokens.append(token_id)
            inputs.append(token_id)

        number_text = llm.decode(number_tokens).strip()

        try:
            if parameter_type == "float" or parameter_type == "number":
                value = float(number_text)
            else:
                value = int(float(number_text))

            print("NUMBER:", number_text)
            print("VALUE:", value)
            final_tokens = llm.encode(str(value)).tolist()[0]

            new_token.extend(final_tokens)

            return new_token, inputs

        except ValueError:
            return new_token, inputs

    def ft_string(
        self,
        inputs: list[int],
        new_token: list[int]
    ) -> tuple[list[int], list[int]]:
        for _ in range(30):
            logits = self.llm.get_logits_from_input_ids(inputs)
            logits = torch.tensor(logits)

            predicted_tensor = torch.argmax(logits)
            token_id = predicted_tensor.item()

            token_text = self.llm.decode([token_id])

            if '"' in token_text and "\n" in token_text:
                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                break
            new_token.append(token_id)
            inputs.append(token_id)

            if '"' in token_text:
                break
        return new_token, inputs

    def generate_text(
        self,
        prompt: str,
        llm: Small_LLM_Model,
        user_request: str
    ) -> str | None:
        used_parameters = []
        inputs = llm.encode(prompt)
        inputs = inputs.tolist()[0]
        new_token = []
        start = 0
        for _ in range(60):
            if start == 0:
                new_token, inputs = self.ft_constrain("{", inputs, new_token)
                start += 1
            elif start == 1:
                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                new_token, inputs = (
                    self.ft_constrain_tokens(
                        self.fixed_tokens["prompt"], inputs, new_token))
                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                new_token, inputs = self.ft_constrain(':', inputs, new_token)
                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                escaped_request = json.dumps(user_request)[1:-1]
                prompt_tokens = llm.encode(escaped_request).tolist()[0]

                new_token, inputs = (
                    self.ft_constrain_tokens(prompt_tokens, inputs, new_token))

                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                new_token, inputs = self.ft_constrain(',', inputs, new_token)

                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                new_token, inputs = (
                    self.ft_constrain_tokens(
                        self.fixed_tokens["name"], inputs, new_token))
                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                start += 1
            elif start == 2:
                new_token, inputs = self.ft_constrain(":", inputs, new_token)
                start += 1
            elif start == 3:
                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                new_token, inputs, name_of_func = (
                    self.ft_constrain_name_function(inputs, new_token, llm))
                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                start += 1
            elif start == 4:
                new_token, inputs = self.ft_constrain(',', inputs, new_token)
                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                new_token, inputs = (
                    self.ft_constrain_tokens(
                        self.fixed_tokens["parameters"], inputs, new_token))
                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                new_token, inputs = self.ft_constrain(':', inputs, new_token)
                start += 1
            elif start == 5:
                new_token, inputs = self.ft_constrain("{", inputs, new_token)
                function_name = llm.decode(name_of_func).strip()
                start += 1
            elif start == 6:
                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                new_token, inputs, parameter = self.ft_constrain_parameters(
                    inputs, new_token, name_of_func, llm)

                new_token, inputs = self.ft_constrain('"', inputs, new_token)
                new_token, inputs = self.ft_constrain(':', inputs, new_token)
                start += 1
            elif start == 7:
                parameters_t = llm.decode(parameter).strip()
                function_name = llm.decode(name_of_func).strip()

                used_parameters.append(parameters_t)

                parameter_type = self.parameter_type_func(
                    function_name, parameters_t)
                if (
                    parameter_type == "number"
                    or parameter_type == "integer"
                    or parameter_type == "float"
                ):
                    new_token, inputs = self.ft_numb_num(
                        inputs, new_token, llm, parameter_type)

                elif parameter_type == "string" or parameter_type == "boolean":
                    new_token, inputs = self.ft_constrain(
                        '"', inputs, new_token)
                    new_token, inputs = self.ft_string(inputs, new_token)

                else:
                    print("Unknown parameter:", parameters_t)
                    return None

                start += 1

            elif start == 8:
                function_name = llm.decode(name_of_func).strip()
                parameters = self.get_parameters(function_name)

                if len(used_parameters) >= len(parameters):
                    new_token, inputs = self.ft_constrain(
                        "}", inputs, new_token)
                    new_token, inputs = self.ft_constrain(
                        "}", inputs, new_token)
                    start = 11
                    continue

                logits = torch.tensor(llm.get_logits_from_input_ids(inputs))
                comma = self.vocab[","]
                brace = self.vocab["}"]

                original_logits = logits.clone()
                logits[:] = float("-inf")
                logits[brace] = original_logits[brace]
                logits[comma] = original_logits[comma]
                predicted_tensor = torch.argmax(logits)
                token = llm.decode([predicted_tensor.item()])
                start += 1
            elif start == 9:
                # print("it reached here the state 9")
                nw = llm.decode(new_token)
                if token == ",":
                    new_token, inputs = self.ft_constrain(
                        ",", inputs, new_token)
                    start = 6
                elif token == "}" and nw[-1] == ",":
                    start = 6
                elif token == "}":
                    start += 1
                else:
                    print("Invalid separator:", token)
                    break
            elif start == 10:
                new_token, inputs = self.ft_constrain("}", inputs, new_token)
                new_token, inputs = self.ft_constrain("}", inputs, new_token)
                start += 1
            else:
                break
        answer = llm.decode(new_token)
        print(repr(answer))
        return answer


def read_vocab(llm: Small_LLM_Model) -> dict[str, int]:
    path = llm.get_path_to_vocab_file()
    try:
        with open(path, "r") as file:
            vocab = json.load(file)
    except FileNotFoundError as e:
        print("Error:", e)
        sys.exit(0)
    return vocab


def write_output(output_text: list[Any]) -> None:
    output_path = Path("data/answers.json")
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


def maaain_t() -> None:
    args = parser_args()

    answer_lst = []
    llm = Small_LLM_Model()
    vocab = read_vocab(llm)
    S = LLM(llm, vocab)

    with open(args.input, "r", encoding="utf-8") as file:
        content = file.read()
        prompt = json.loads(content)

    with open(args.functions_definition, "r", encoding="utf-8") as file:
        functions_text = file.read()
    for i in range(11):
        print("--------------->the promopt", i)
        user_request = prompt[i]["prompt"]
        answer = S.generate_text(f"""You are a function-calling assistant.

        You are given:

        1. A list of available functions in JSON format.
        2. A user's request.

        Your task is to determine:
        - which function should be called,
        - and what arguments should be passed to it.

        Available Functions:

        {functions_text}

        ----------------------------------------

        User Request:

        {user_request}

        ----------------------------------------

        {{
        "function": "<function_name>",
        "arguments": {{
            ...
        }}
        }}

        Do not explain your reasoning.
        Do not return Markdown.
        If no function matches, return null.""", llm, user_request)
        if answer is not None:
            try:
                answer_lst.append(json.loads(answer))
            except json.JSONDecodeError as e:
                print("INVALID JSON:")
                print(repr(answer))
                print("ERROR:", e)
    write_output(answer_lst)


def main_logic() -> None:
    try:
        maaain_t()
    except KeyboardInterrupt as e:
        print("Error:", e)
        sys.exit(0)


main_logic()
