from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from client.chain_base import ChainDirector, ConcreteChainBase


BASE_SYSTEM_PROMPT_FORMAT = (
    "You are a helpful instruction rewriter.\n"
    "1. Follow the prompts and rewrite from #Instruction# to #Evolved instruction#.\n"
    "2. In the next step, the #Instruction# is replaced by the #Evolved instruction# received in '1.' to prepare for further evolution.\n\n"
    "Repeat process '1.' to '2.' above {evol_times} times to generate a multiple structured list containing {evol_times} sets of #Evolved instruction# and #Stage#."
    " Strictly control the number of iterations, and the output number of #Stage# must not be greater than {evol_times}.\n"
)

INITIAL_METHOD_PROMPT = """You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version. Please follow the steps below to rewrite the given "#Instruction#" into a more complex version.

Step 1: Please read the "#Instruction#" carefully and list all the possible methods to make this instruction more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Please do not provide methods to change the language of the instruction!

Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# more complex. The plan should include several methods from the #Methods List#.

Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only add 10 to 20 words into the "#Instruction#".

Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#. Just provide the #Finally Rewritten Instruction# without any explanation.

Please reply strictly in the following format:

Step 1 #Methods List#:
Step 2 #Plan#: 
Step 3 #Rewritten Instruction#:
Step 4 #Finally Rewritten Instruction#:
"""

END_OF_METHOD = "\n\n#Instruction#:\n{instruction}"


class EvolvedInstruction(BaseModel):
    """ Evolved instruction"""
    stage: int = Field(description="Serial number of the evolution stage.")
    evolved_instruction: str = Field(description="Evolved instruction.")


class EvolvedList(BaseModel):
    """List of evolved instructions."""
    transition: List[EvolvedInstruction] = Field(description="A list containing multiple evolved instructions.")


class InstructEvolveGenerator(ConcreteChainBase):
    def __init__(
            self,
            chat_model: BaseChatModel,
            evol_method: str = INITIAL_METHOD_PROMPT,
            evol_times: int = 10,
            use_evol_times_check: bool = False,
            add_original_instruction: bool = False,
    ):
        """
        Parameters
        ----------
        chat_model
        evol_method
        evol_times
        """

        super().__init__()
        self._llm = chat_model
        self._evol_method = evol_method
        self._evol_times = evol_times
        self._use_check = use_evol_times_check
        self._add_inst = add_original_instruction

    def _create_chain_director(
            self,
            director_config: Optional[Dict],
    ) -> ChainDirector:
        system_prompt = BASE_SYSTEM_PROMPT_FORMAT.format(evol_times=self._evol_times)
        human_prompt = self._evol_method + END_OF_METHOD

        return ChainDirector(
            chat_model=self._llm,
            system_prompt=system_prompt,
            human_prompt=human_prompt,  # {instruction}
            struct_type=EvolvedList,
        )

    def _invoke_handling(
            self,
            input: Dict[Literal["instruction"], str],
            **kwargs
    ) -> List[Dict[str, Any]]:

        if self._use_check:
            res = self._inst_evol_num_check(input, **kwargs)
        else:
            res = self._chain_director.invoke(
                input,  # {instruction}
                **kwargs,
            )

        if self._add_inst:
            org = EvolvedInstruction(**{'stage': 0, 'evolved_instruction': input['instruction']})
            res.transition.insert(0, org)

        res = res.model_dump()
        # return res.model_dump()

        res_list = res['transition']
        return res_list

    def _inst_evol_num_check(
            self,
            input: Dict[Literal["instruction"], str],
            **kwargs
    ):
        res = self._chain_director.invoke(input, **kwargs)
        if not res:
            return self._inst_evol_num_check(input, **kwargs)
        # 要素の数が想定した個数になっているかチェックするためにListの中身を確認
        trans_list = res.transition
        trans_list = [e for e in trans_list if e.evolved_instruction]  # 空文字は削除
        return self._inst_evol_num_check(input, **kwargs) if len(trans_list) != self._evol_times else res


if __name__ == "__main__":
    """
    python -m client.concrete.inst_evol_gen
    """

    from model.groq_llm import GroqChatBase

    llm = GroqChatBase(
        model_name="llama-3.1-70b-versatile",
        # model_name="mixtral-8x7b-32768",
        requests_per_second=0.32,
        temperature=0.8
    )

    # test = "1+1=?"
    test = "Hans booked a room in a hotel. The hotel has 10 floors with 10 identical rooms on each floor. Because of an accident, the last floor is unavailable for the guests. Considering there are no other guests, in how many different rooms could Hans be checked in?"

    method = INITIAL_METHOD_PROMPT

    gen = InstructEvolveGenerator(
        chat_model=llm,
        evol_method=method,
        evol_times=10,
        use_evol_times_check=True,
        add_original_instruction=True,
    )

    inst = {
        "instruction": test,
    }

    result = gen(inst)
    print(result)
    # for r in result:
    #     print(r['evolved_instruction'])

