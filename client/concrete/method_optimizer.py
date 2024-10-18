from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from client.chain_base import ChainDirector, ConcreteChainBase

BASE_SYSTEM_PROMPT = (
    "You are a helpful optimizer that improves the methods of giving prompts to GPT AI.\n"
    "I will provide you with the #Method prompt# for evolving the above instructions,"
    " and the #Feedback# from the evolution failure case.\n"
)

PROMPT_BASE = (
    "You need to optimize the #Method prompt# based on the #Feedback# from the evolution failure case,"
    " without harming the performance on other cases,"
    " and ensure that the complexity increase brought by the optimized method is not lower than the previous method.\n"
    " Please provide #Optimized method# in structured format.\n\n"
    "#Feedback#:\n{feedback}\n\n"
    "#Method prompt#:\n{method_prompt}\n\n"
)


class OptimizedMethod(BaseModel):
    """ The optimised method prompt GPT AI to evolve the instructions. """
    optimized_method: str = Field(description="The optimized method prompt.")


class EvolveMethodOptimizer(ConcreteChainBase):
    def __init__(
            self,
            chat_model: BaseChatModel,
    ):
        """
        :param chat_model:
        """
        super().__init__()
        self._llm = chat_model

    def _create_chain_director(
            self,
            director_config: Optional[Dict],
    ) -> ChainDirector:
        system_prompt = BASE_SYSTEM_PROMPT
        human_prompt = PROMPT_BASE

        return ChainDirector(
            chat_model=self._llm,
            system_prompt=system_prompt,
            human_prompt=human_prompt,  # {feedback}, {method_prompt}
            struct_type=OptimizedMethod,
        )

    def _invoke_handling(
            self,
            input: Dict[Literal["feedback", "method_prompt"], str],
            **kwargs
    ) -> str:
        res = self._chain_director.invoke(
            input,  # {feedback}, {method_prompt}
            **kwargs,
        )

        res = res.model_dump()
        # return res.model_dump()

        res_list = res['optimized_method']
        return res_list


if __name__ == "__main__":
    """
    python -m client.concrete.method_optimizer
    """

    from model.groq_llm import GroqChatBase

    llm = GroqChatBase(
        model_name="llama-3.1-70b-versatile",
        # model_name="mixtral-8x7b-32768",
        requests_per_second=0.32,
        temperature=0.6
    )

    initial_method = """
    You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version.
    Please follow the steps below to rewrite the given "#Instruction#" into a more complex version.

    Step 1: Please read the "#Instruction#" carefully and list all the possible methods to make this instruction more complex (to 
    make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Please do not provide methods to 
    change the language of the instruction!

    Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# more 
    complex. The plan should include several methods from the #Methods List#.

    Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only add 10 to 
    20 words into the "#Instruction#".

    Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. Ensure that the #Rewritten 
    Instruction# is only a more complex version of the #Instruction#. Just provide the #Finally Rewritten Instruction# without any 
    explanation.

    Please reply strictly in the following format:

    Step 1 #Methods List#:
    Step 2 #Plan#: 
    Step 3 #Rewritten Instruction#:
    Step 4 #Finally Rewritten Instruction#:
    """

    # feedback = 'Lack of explicit mathematical operation.'
    feedback = 'Fails to enhance the complexity of the instruction.'

    gen = EvolveMethodOptimizer(chat_model=llm)

    inst = {
        "feedback": feedback,
        "method_prompt": initial_method,
    }

    result = gen(inst)
    print(result)

    '''
    Step 1: Thoroughly analyze the given "#Instruction#" and compile a detailed list of potential methods to enhance its complexity. The aim is to create challenges that would make it more difficult for well-known AI models such as ChatGPT and GPT-4 to interpret accurately. Ensure that the methods do not involve altering the language of the instruction itself.

    Step 2: Develop a comprehensive strategy utilizing the #Methods List# from Step 1. This plan should incorporate multiple techniques from the #Methods List# to significantly elevate the complexity of the #Instruction#.

    Step 3: Systematically implement the plan, documenting each stage, and present the #Rewritten Instruction#. The revised instruction should expand by 10 to 20 words, ensuring the complexity is notably increased.

    Step 4: Conduct a meticulous review of the #Rewritten Instruction# to pinpoint any elements that may seem unreasonable. Confirm that the #Rewritten Instruction# represents a genuinely more complex iteration of the original #Instruction#. Provide only the #Finally Rewritten Instruction# without additional commentary.

    Please respond strictly in the following format:

    Step 1 #Methods List#:
    Step 2 #Plan#: 
    Step 3 #Rewritten Instruction#:
    Step 4 #Finally Rewritten Instruction#: 
    '''
