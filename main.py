import argparse
import random

import numpy as np

from auto_evol.manager import AutoEvolveConfig, AutoEvolveManager
from client.concrete.inst_evol_gen import INITIAL_METHOD_PROMPT


def fix_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


INPUT_FILE = 'data/gsm8k_en_pieces.jsonl'
OUTPUT_DIR = './data/output'
RANDOM_SEED = 0


def parse_option():
    dc = """
    """
    parser = argparse.ArgumentParser(description=dc)

    parser.add_argument('--input_file', type=str, dest='input_file', default=INPUT_FILE,
                        help=f""" A *.jsonl file containing seed instructions. e.g. {INPUT_FILE}""")
    parser.add_argument('--output_dir', type=str, dest='output_dir', default=OUTPUT_DIR,
                        help=f""" output directory. e.g. {OUTPUT_DIR}""")
    parser.add_argument("--num_instructions_for_train", type=int, default=7,
                        help="Number of instructions used to extruct feedbacks.", )
    parser.add_argument("--num_instructions_for_test", type=int, default=3,
                        help="Number of instructions used to evaluate the optimised methods.", )
    parser.add_argument("--num_of_evolve", type=int, default=10,
                        help="Number of evolutions for each instruction.", )
    parser.add_argument("--num_of_generation", type=int, default=10,
                        help="Number of generations to optimise the methods.", )
    return parser.parse_args()


def main():
    fix_seeds(RANDOM_SEED)

    from model.groq_llm import GroqChatBase
    llm_main = GroqChatBase(
        model_name="llama-3.1-70b-versatile",
        # requests_per_second=0.05,  # 1 request every 10 seconds or so.
        requests_per_second=0.07,
        # max_tokens=2048,
        temperature=0.4,
    )

    # from model.azure_openai_llm import AzureChatBase
    # llm_main = AzureChatBase(
    #     model_name="gpt-4o-mini",
    #     api_version="2024-05-01-preview",
    #     temperature=0.6,
    #     # max_tokens=2048,
    #     requests_per_second=0.32,
    # )


    aem = AutoEvolveManager(
        initial_target_method_prompt=INITIAL_METHOD_PROMPT,
        main_llm=llm_main,
        # num_of_feedbacks_integration=10,  # Threshold for whether to reduce the number of feedbacks.
        num_of_feedbacks_integration=5,  # Threshold for whether to reduce the number of feedbacks.
    )

    args = parse_option()
    evol_cfg = AutoEvolveConfig(**vars(args))
    aem(evol_cfg)


if __name__ == "__main__":
    """
    python main.py \
	--input_file "./data/gsm8k_en_pieces.jsonl" \
	--output_dir "./data/output" \
	--num_instructions_for_train 3 \
	--num_instructions_for_test 2 \
	--num_of_evolve 10 \
	--num_of_generation 3 
    """
    main()
