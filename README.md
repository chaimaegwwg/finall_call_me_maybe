*This project has been created as part of the 42 curriculum by cramadan*

# Function Calling LLM

## Description

This project implements a small function-calling system using a local Large Language Model (LLM). The program takes a user's natural-language request and generates a structured JSON object containing the function to call and its arguments.

The project uses a Hugging Face causal language model through the `Small_LLM_Model` SDK. Instead of allowing the model to freely generate the complete JSON, the implementation uses **constrained decoding** to control which tokens can be generated at each step.

The main goals are to:

* Load function definitions from a JSON file.
* Understand a user's request.
* Select the appropriate function.
* Select valid parameters for that function.
* Generate parameter values according to their expected types.
* Produce valid JSON that can be processed programmatically.

## Features

* Local LLM inference.
* Function selection using constrained token generation.
* Parameter-name selection based on the function definition.
* Support for `string`, `number`, `integer`, `float`, and `boolean` parameters.
* JSON escaping for user requests.
* Command-line arguments for input, output, and function-definition files.
* Automatic JSON validation of generated results.

## Instructions

### Requirements

The project requires Python and the dependencies defined by the project environment.

The main libraries used are:

* `torch`
* `transformers`
* `huggingface_hub`
* `llm_sdk`

### Installation

Install the project dependencies using the project's configured environment, for example:

```bash
uv sync
```

Make sure the required Hugging Face model can be downloaded before running the program.

### Input files

The default files are:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
```

`functions_definition.json` contains the available functions and their parameters.

`function_calling_tests.json` contains user requests used to test the function-calling system.

### Execution

Run the program with:

```bash
uv run -m src
```

Custom input files can be provided with:

```bash
uv run -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

The generated results are written to the specified output JSON file.

## Algorithm Explanation

The core of the project is a **constrained decoding algorithm**.

Normally, a language model can choose any token from its vocabulary at every generation step. This can result in invalid function names, invalid parameter names, incorrect JSON syntax, or unwanted additional text.

This implementation restricts the possible tokens depending on the current generation state.

The generation process is divided into several states:

1. **Start JSON object**

   * The decoder forces `{`.

2. **Generate the `prompt` field**

   * The required JSON syntax and the `prompt` key are inserted directly.
   * The user's request is escaped with `json.dumps()` before being inserted.

3. **Generate the `name` field**

   * The model must select one of the functions defined in `functions_definition.json`.
   * The token sequences of all available function names are compared.
   * At every step, only tokens that can continue at least one valid function name are allowed.
   * Functions that no longer match the generated sequence are removed from the candidates.

4. **Generate the `parameters` field**

   * After the function is selected, its parameter definitions are retrieved.
   * The same constrained decoding technique is used to select a valid parameter name.

5. **Generate the parameter value**

   * The expected parameter type is obtained from the function definition.
   * Numeric values are generated as tokens and converted to the required numeric type.
   * String and boolean values are generated using string constraints.

6. **Generate the remaining parameters**

   * The decoder checks whether more parameters are required.
   * It restricts the next token to either `,` or `}`.
   * A comma causes the decoder to select another parameter.
   * A closing brace finishes the parameters object.

7. **Finish the JSON object**

   * The final `}` characters are constrained explicitly.
   * The generated token sequence is decoded into a JSON string.

The implementation therefore combines **model prediction** with **hard token constraints**. The model decides between valid choices, while the decoder prevents many syntactically or semantically invalid choices.

## Design Decisions

A state-based generation system was chosen because different parts of the JSON structure require different constraints.

The implementation separates the generation logic into functions such as:

* `ft_constrain()` for forcing a specific token.
* `ft_constrain_tokens()` for forcing a sequence of tokens.
* `ft_constrain_name_function()` for selecting a valid function name.
* `ft_constrain_parameters()` for selecting valid parameter names.
* `ft_numb_num()` for generating numeric values.
* `ft_string()` for generating string values.

The function definitions are kept in JSON instead of being hardcoded in the generation algorithm. This makes the system easier to extend with new functions.

The vocabulary is also loaded dynamically from the LLM SDK so that token IDs do not have to be manually hardcoded.

## Performance Analysis

### Accuracy

Constrained decoding improves reliability compared with completely free generation because important parts of the output are restricted.

Function names and parameter names can only come from the definitions provided to the program. JSON syntax is also explicitly controlled for important structural tokens such as `{`, `}`, `"`, `:`, and `,`.

### Speed

The main performance cost comes from repeatedly calling the model to obtain logits for each generated token.

For every constrained generation step, the program performs another forward pass through the model. This makes the approach slower than generating the entire sequence freely with standard generation methods.

However, the project prioritizes correctness and control over maximum generation speed.

### Reliability

The output is checked using Python's `json.loads()` before being added to the final results. Invalid JSON is therefore detected instead of silently being written as a valid result.

The constrained approach also reduces common generation errors such as:

* Unknown function names.
* Unknown parameter names.
* Invalid JSON structure.
* Additional explanatory text.
* Incorrect separators.

## Challenges Faced

One of the main challenges was controlling the model after it had generated the expected JSON. A causal language model does not automatically know that it should stop exactly at the end of the desired structure, so additional tokens could be generated.

Another challenge was handling tokenization. A function name or parameter name may consist of multiple tokens, so constraining only the first token is not sufficient. The implementation therefore compares complete token sequences and removes candidates that no longer match.

Numeric values also required special handling because the model generates tokens rather than Python numbers. The generated tokens are decoded into text and then converted to `float` or `int` depending on the parameter definition.

JSON escaping was another issue. User requests can contain characters such as quotes that would make the generated JSON invalid. The implementation uses `json.dumps()` to escape the request before inserting it into the generated JSON.

## Testing Strategy

The implementation is tested using the test cases stored in:

```text
data/input/function_calling_tests.json
```

For each test:

1. The user's request is extracted.
2. The LLM generates a constrained function call.
3. The generated result is printed for debugging.
4. `json.loads()` is used to verify that the result is valid JSON.
5. Valid results are stored in the output file.

Example test cases include arithmetic requests such as:

```text
What is the product of 3 and 5?
```

The expected structure is similar to:

```json
{
    "prompt": "What is the product of 3 and 5?",
    "name": "fn_multiply_numbers",
    "arguments": {
        "a": 3.0,
        "b": 5.0
    }
}
```

Testing also focused on edge cases involving numeric conversion, strings, JSON escaping, function selection, parameter selection, and stopping generation at the correct location.

## Example Usage

Run the default test set:

```bash
uv run -m src
```

Run with custom files:

```bash
uv run -m src \
    --functions_definition path/to/functions_definition.json \
    --input path/to/function_calling_tests.json \
    --output path/to/results.json
```

Example input:

```json
{
    "prompt": "What is the product of 3 and 5?"
}
```

Example output:

```json
{
    "prompt": "What is the product of 3 and 5?",
    "name": "fn_multiply_numbers",
    "arguments": {
        "a": 3.0,
        "b": 5.0
    }
}
```

## Project Structure

```text
.
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/
│       └── function_calling_results.json
├── llm_sdk/
│   └── llm_sdk.py
├── src/
│   └── ...
└── README.md
```

## Resources

Useful references for this project include:

* Hugging Face Transformers documentation: https://huggingface.co/docs/transformers/
* PyTorch documentation: https://pytorch.org/docs/
* Hugging Face Qwen models: https://huggingface.co/Qwen
* Python `json` documentation: https://docs.python.org/3/library/json.html
* Python `argparse` documentation: https://docs.python.org/3/library/argparse.html
* JSON specification: https://www.json.org/

### AI Usage

AI tools were used as a development and debugging aid during this project.

They were used for:

* Understanding Hugging Face Transformers and causal language models deeply.
* Understanding logits and token prediction.
* Understanding constrained decoding techniques.
