import os.path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, final, Union, Any
import copy
import random

from tqdm.auto import tqdm
from langchain_core.language_models import BaseChatModel

from util.file_tools import JsonHandler, JsonlHandler
from client.concrete.inst_evol_gen import INITIAL_METHOD_PROMPT
from auto_evol.director import AutoEvolveDirector


@dataclass
class AutoEvolveConfig:
    input_file: str
    output_dir: str
    num_instructions_for_train: int
    num_instructions_for_test: int
    num_of_evolve: int
    num_of_generation: int


class AutoEvolveManager(object):
    def __init__(
            self,
            main_llm: BaseChatModel,
            optim_llm: Optional[BaseChatModel] = None,
            anal_llm: Optional[BaseChatModel] = None,
            initial_target_method_prompt: str = INITIAL_METHOD_PROMPT,
            num_of_feedbacks_integration: int = 8,
    ):
        """
        Parameters
        ----------
        initial_target_method_prompt: str
            Initial method prompts to be optimised.
        main_llm: BaseChatModel
            A main model for evolving instructions.
        optim_llm: Optional[BaseChatModel]
            The model for optimizing method prompt. The main model is used if not specified.
        anal_llm: Optional[BaseChatModel]
            The model for analysing evolutionary trajectories. The main model is used if not specified.
        num_of_feedbacks_integration: int
            Threshold for whether to reduce the number of feedbacks.
        """
        self._target_method = initial_target_method_prompt
        self._model = main_llm
        self._ae_director = AutoEvolveDirector(
            main_llm,
            optim_llm,
            anal_llm,
            num_of_feedbacks_integration,
        )

    def __call__(
            self,
            cfg: AutoEvolveConfig
    ) -> None:
        return self.file_handling(cfg)

    def file_handling(
            self,
            cfg: AutoEvolveConfig
    ) -> None:
        input_file = cfg.input_file
        output_dir = cfg.output_dir
        num_inst_for_train = cfg.num_instructions_for_train
        num_inst_for_test = cfg.num_instructions_for_test
        num_of_evol = cfg.num_of_evolve
        mun_of_gent = cfg.num_of_generation

        # output directory check
        os.makedirs(output_dir, exist_ok=True)

        # file handle tools
        jh = JsonHandler()  # tool for *.json files.
        jlh = JsonlHandler()  # tool for *.jesonl files.

        objs = jlh.read(input_file)
        # print(len(objs))

        org_inst = [d['query'] for d in objs]
        # print(len(objs))

        num_of_demands = num_inst_for_train + num_inst_for_test
        assert (len(org_inst) > num_of_demands), f"Instructions included in {input_file} are too few."

        inst_pod = copy.deepcopy(org_inst)
        # print(len(inst_pod))

        seed_inst = random.choices(inst_pod, k=num_inst_for_train)
        inst_pod = [e for e in inst_pod if e not in seed_inst]
        # print(len(inst_pod))

        print('----- seed instructions -----')
        for i in seed_inst:
            print(i)
        print('-----------------------------')

        method_prompt = self._target_method
        for evol_i in tqdm(range(mun_of_gent), desc='Generation Stage : '):
            # seed_inst = random.choices(inst_pod, k=train_seed_num)
            # inst_pod = [e for e in inst_pod if e not in seed_inst]

            eval_inst = random.choices(inst_pod, k=num_inst_for_test)

            print('-------------target method----------------')
            print(method_prompt)
            print('------------------------------------------')
            method, score = self._ae_director.method_optimizer(
                target_method=method_prompt,
                instructions_for_seed=seed_inst,
                instructions_for_eval=eval_inst,
                num_instructions_to_evolve=num_of_evol,
            )

            method_prompt = method

            state = {f"gen_{evol_i}": {"method": method, "score": score}}

            out_file = f"{output_dir}/gen_{evol_i + 1}.json"
            jh.write(state, out_file)


if __name__ == "__main__":
    """
    python -m auto_evol.manager
    """

    from model.groq_llm import GroqChatBase

    llm_main = GroqChatBase(
        model_name="llama-3.1-70b-versatile",
        # requests_per_second=0.32,
        # requests_per_second=0.16,
        max_tokens=2048,
        requests_per_second=0.08,
        temperature=0.6,
    )

    # llm_optim = GroqChatBase(
    #     model_name="llama-3.1-70b-versatile",
    #     # requests_per_second=0.32,
    #     # requests_per_second=0.16,
    #     requests_per_second=0.08,
    #     temperature=0.8,
    # )


    test_cfg = dict(
        input_file="./data/gsm8k_en_pieces.jsonl",
        output_dir="./data/output",
        num_instructions_for_train=7,
        num_instructions_for_test=3,
        num_of_evolve=10,
        num_of_generation=10,
    )

    em = AutoEvolveManager(
        initial_target_method_prompt=INITIAL_METHOD_PROMPT,
        main_llm=llm_main,
        # optim_llm=llm_optim,
        num_of_feedbacks_integration=8,  # Threshold for whether to reduce the number of feedbacks.
    )

    em(AutoEvolveConfig(**test_cfg))


