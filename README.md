# Auto-Evolve-Instruct-like

*Source: Adaptation of [Automatic Instruction Evolving for Large Language Models](https://arxiv.org/abs/2406.00770)*

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/tana114/Auto-Evolve-Instruct-like.git
   cd Auto-Evolve-Instruct-like
   ```

2. Set up your environment variables:
   Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

For example, run main.py as follows:
```
python main.py \
--input_file "./data/gsm8k_en_pieces.jsonl" \
--output_dir "./data/output" \
--num_instructions_for_train 3 \
--num_instructions_for_test 2 \
--num_of_evolve 10 \
--num_of_generation 3 
```

However, the behaviour is not stable when using `groq` so please consider using `openai`, etc.

When checking the function of each module, it is easier to understand the behaviour if you run the module by itself, as shown below.
```
python -m model.groq_llm
```

```
python -m client.concrete.trajectory_analyser
```



