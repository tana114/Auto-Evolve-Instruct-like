import json
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, Union

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from client.chain_base import ChainDirector, ConcreteChainBase

BASE_SYSTEM_PROMPT = """You are a helpful assistant that analyses evolutionary transitions in instructions.
- Follow the prompts and identify cases that failed to evolve, and provide their case #Id# and #Reason#.

Generate a multiple structured list containing sets of  #Evolved instruction# and #Id#.
"""

PROMPT_BASE = (
    "The following list shows cases where an Instruction evolves into a more complex version of an Instruction.\n"
    "For each case, stage 0 represents the Instruction in its initial state,"
    " and each subsequent stage requires an increase in complexity based on the previous stage.\n"
    "Please identify cases that failed to evolve, and provide their case #Id# and #Reason#.\n"
    "Only the reasons should be stated in the #Reason#, without any reference to the target stage.\n"
    "Do not include specific case-related information in the #Reason#, but describe universal themes that point to more fundamental issues.\n"
    "All responses must be output in English.\n\n"
    "{evolutionary_trajectory}"
)

class EvolveFailure(BaseModel):
    """ Evolutionary failure analysis results """
    id: int = Field(description="the unique serial number.")
    reason: str = Field(description="Reasons for being judged an evolutionary failure.")


class FeedbackList(BaseModel):
    """List of evolutionary failure analysis results."""
    feedbacks: List[EvolveFailure] = Field(
        description="A list containing multiple evolutionary failure analysis results.")


class EvolveTrajectoryAnalyser(ConcreteChainBase):
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
            human_prompt=human_prompt,  # {evolutionary_trajectory}
            struct_type=FeedbackList,
        )

    def _invoke_handling(
            self,
            input: Dict[Literal["evolutionary_trajectory"], str],
            **kwargs
    )  -> List[Dict[str, Any]]:
        res = self._chain_director.invoke(
            input,  # {evolutionary_trajectory}
            **kwargs,
        )

        res = res.model_dump()
        # return res.model_dump()

        res_list = res['feedbacks']
        return res_list

    @staticmethod
    def dict2prompt(dict_list:Union[Dict, List[Dict]])->str:
        dict_list = [dict_list] if isinstance(dict_list, (str, Dict)) else dict_list
        prompt = ""
        for d in dict_list:
            prompt += json.dumps(d, indent=2, ensure_ascii=False)
            prompt += "\n"
        return prompt


if __name__ == "__main__":
    """
    python -m client.concrete.trajectory_analyser
    """


    from model.groq_llm import GroqChatBase

    llm = GroqChatBase(
        model_name="llama-3.1-70b-versatile",
        # model_name="mixtral-8x7b-32768",
        requests_per_second=0.32,
        temperature=0.4
    )

    trans_dict = {'transition':
        [
            {'stage': 0,
             'evolved_instruction': 'Natalia sold clips to 48of her friends in April,and then she sold half as many clips in May.How many clips didNatalia sell altogether in April and May?'},
            {'stage': 1,
             'evolved_instruction': 'In the month of April, Natalia successfully sold a total of 48 clips to a diverse group of her friends, and subsequently, in May, the quantity sold diminished to half of that amount. Can you calculate the cumulative total of clips that Natalia sold across both April and May?'},
            {'stage': 2,
             'evolved_instruction': 'During the month of April, Natalia engaged in selling a total of 48 clips to her circle of friends, and for the subsequent month of May, she managed to sell precisely half that number of clips. Could you determine the overall total of clips sold by Natalia when combining her sales from both April and May?'},
            {'stage': 3,
             'evolved_instruction': 'In April, Natalia actively marketed and sold 48 clips to a considerable number of her friends, and in the following month of May, her sales figures fell to exactly half of what she achieved in April. What is the total number of clips that Natalia sold when aggregating the sales from both months?'},
            {'stage': 4,
             'evolved_instruction': "In the month of April, Natalia executed a successful sales strategy that led her to sell a total of 48 clips to a significant number of her friends; however, in May, her sales plummeted to precisely half of the previous month's total. Can you ascertain the overall total of clips that Natalia sold when summing her sales from both April and May?"},
            {'stage': 5,
             'evolved_instruction': 'In April, Natalia adeptly sold 48 clips to a wide array of her friends, but in the subsequent month of May, the quantity sold was reduced to exactly half of her April sales. What is the grand total of clips that Natalia sold when considering both months of April and May?'},
            {'stage': 6,
             'evolved_instruction': 'In April, Natalia successfully sold 48 clips to a diverse selection of friends, only to see her sales decrease to half that amount in May. Could you compute the total number of clips sold by Natalia over the two months, April and May?'},
            {'stage': 7,
             'evolved_instruction': 'Throughout the month of April, Natalia managed to sell a notable total of 48 clips to her network of friends, but come May, she experienced a decline in sales, achieving only half of her previous total. What is the complete total of clips Natalia sold when you consider both months combined?'},
            {'stage': 8,
             'evolved_instruction': 'In the month of April, Natalia achieved impressive sales by offering 48 clips to her friends, but in May, her sales dropped to a mere half of that amount. What is the overall total of clips that she sold when accounting for both April and May?'},
            {'stage': 9,
             'evolved_instruction': 'During April, Natalia succeeded in selling 48 clips to her friends; however, in May, her sales figures reflected a reduction, specifically amounting to half of her April performance. How many clips did Natalia manage to sell in total when both months are considered?'},
            {'stage': 10,
             'evolved_instruction': "In April, Natalia effectively sold a total of 48 clips to her assortment of friends, yet in May, her numbers were reduced to precisely half of the previous month's sales. What is the overall total of clips sold by Natalia when aggregating figures from both April and May?"}
        ]}

    trans = trans_dict.get('transition')

    gen = EvolveTrajectoryAnalyser(chat_model=llm)

    trans_prompt = gen.dict2prompt(trans)
    # print(trans_prompt)

    inst = {
        "evolutionary_trajectory": trans_prompt,
    }

    result = gen(inst)
    print(result)


