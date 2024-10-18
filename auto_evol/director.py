import sys
from typing import List, Dict, Tuple, Optional, final, Union, Any
import numpy as np
from tqdm.auto import tqdm
from langchain_core.language_models import BaseChatModel

from client.concrete.inst_evol_gen import InstructEvolveGenerator
from client.concrete.trajectory_analyser import EvolveTrajectoryAnalyser
from client.concrete.method_optimizer import EvolveMethodOptimizer
from util.sentence_clustering import SentenceClustering


class AutoEvolveDirector(object):
    def __init__(
            self,
            main_llm: BaseChatModel,
            optim_llm: Optional[BaseChatModel] = None,
            anal_llm: Optional[BaseChatModel] = None,
            num_of_feedbacks_integration: Optional[int] = 8,
    ):
        """

        Parameters
        ----------
        main_llm: BaseChatModel
            A main model for evolving instructions.
        optim_llm: Optional[BaseChatModel]
            The model for optimizing method prompt. The main model is used if not specified.
        anal_llm: Optional[BaseChatModel]
            The model for analysing evolutionary trajectories. The main model is used if not specified.
        num_of_feedbacks_integration: Optional[int]
            Threshold for whether to reduce the number of feedbacks.
        """
        self._evol_llm = main_llm
        self._optim_llm = main_llm if not optim_llm else optim_llm
        self._anal_llm = main_llm if not anal_llm else anal_llm
        self._feedbacks_integrate_thresh = num_of_feedbacks_integration if num_of_feedbacks_integration else sys.maxsize
        self._clustering_model_name: str = 'distilbert-base-uncased'

    def evolve_multiple_instructions(
            self,
            instructions: Union[str, List[str]],
            evol_method: str,
            evol_times: int,
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
        # ):
        """
        Generate evolution histories and feedbacks of multiple instructions using the same method prompt.

        Parameters
        ----------
        instructions: List[str]
            the list of instructions to be evolved by evol_method.
        evol_method: str
            the method for evolving the instructions.
        evol_times
            Number of times to evolve each instruction.

        Returns
        -------
            history_list: [List[List[Dict]]
            [
                [{'stage': 0, 'evolved_instruction': '次の数学の問題に...'},
                    ...,
                {'stage': 10, 'evolved_instruction': '...ができます。'}],
                ...,
            ]
            feedback_list: [List[List[Dict]]
            [
                [{'id': 0, 'reason': 'Lack of clear problem statement'},
                    ...,
                {'id': 10, 'reason': 'Insufficient context for problem-solving'}],
                ...,
            ]
        """
        inst_list = [instructions] if isinstance(instructions, str) else instructions

        ''' evolve input instructions '''
        ieg = InstructEvolveGenerator(
            chat_model=self._evol_llm,
            evol_method=evol_method,
            evol_times=evol_times,
            use_evol_times_check=True,
            add_original_instruction=True,
        )

        history_list: List[List[Dict[str, Any]]] = []
        for inst in tqdm(instructions, leave=False, desc='InstructEvolve: '):
            invoke_input = {'instruction': inst}
            res: List[Dict[str, Any]] = ieg(invoke_input)
            history_list.append(res)

        assert all(item is not None for item in inst_list) , ""
        assert len(inst_list) == len(history_list), f"The number of the history_list does not match input instructions."

        ''' evolve trajectory analyse '''
        eta = EvolveTrajectoryAnalyser(chat_model=self._anal_llm)

        feedback_list: List[List[Dict[str, Any]]] = []
        for hist in tqdm(history_list, leave=False, desc='TrajectoryAnalyser: '):
            if not hist:
                print('-----------',history_list)
            trans_prompt = eta.dict2prompt(hist)
            invoke_input = {"evolutionary_trajectory": trans_prompt, }
            res: List[Dict[str, Any]] = eta(invoke_input)
            feedback_list.append(res)

        return history_list, feedback_list

    def integrate_multiple_feedbacks(
            self,
            feedback_list: List[List[Dict[str, Any]]],
            boundary_count: int,
    ) -> List[str]:
        """
        Integrate multiple feedbacks from EvolveTrajectoryAnalyser.

        Parameters
        ----------
        feedback_list: List[List[Dict]]
        [
            [{'id': 0, 'reason': 'Lack of clear problem statement'},
                ...,
            {'id': 10, 'reason': 'Insufficient context for problem-solving'}],
            ...,
        ]

        boundary_count:int
            Threshold for integration or not.

        Returns
        -------
        union_list: List[str]
        """

        # fb_list = [feedback_list] if isinstance(feedback_list, (str,)) else feedback_list
        reasons = [[fd['reason'] for fd in obj] for obj in feedback_list]
        flattened_list = [
            item
            for sublist in reasons
            for item in sublist
        ]

        unique_list = list(set(flattened_list))

        if len(unique_list) > boundary_count:
            sc = SentenceClustering(
                self._clustering_model_name,
                minimum_clusters_num=boundary_count,
            )
            return sc(unique_list)
        return unique_list

    def optimize_method_based_feedbacks(
            self,
            target_method: str,
            feedbacks: List[str],
    ) -> List[str]:
        """
        Optimize the target_method based on feedbacks.
        From a single target_method, a number of optimized methods are generated equal to the number of feedbacks.

        Parameters
        ----------
        target_method
        feedbacks

        Returns
        -------

        """
        evol_methods = []
        for fb in tqdm(feedbacks, leave=False, desc='MethodOptimizer: '):
            emo = EvolveMethodOptimizer(chat_model=self._optim_llm)
            invoke_input = {
                "feedback": fb,
                "method_prompt": target_method
            }
            evol_methods.append(emo(invoke_input))

        evol_methods = [s for s in evol_methods if s]
        fb_num = len(feedbacks)
        assert len(evol_methods) == fb_num, f"The number of the evol_methods does not match the input feedbacks."
        return evol_methods

    def assessing_superior_methods(
            self,
            candidate_methods: Union[str, List[str]],
            instructions_for_eval: Union[str, List[str]],
            num_instructions_to_evolve: int,
    ) -> Tuple[str, Any]:
        mt_list = [candidate_methods] if isinstance(candidate_methods, str) else candidate_methods
        inst_list = [instructions_for_eval] if isinstance(instructions_for_eval, str) else instructions_for_eval

        trj_list = []
        for m in tqdm(mt_list):
            _, fb_list = self.evolve_multiple_instructions(
                instructions=inst_list,
                evol_method=m,
                evol_times=num_instructions_to_evolve,
            )
            trj_list.append(fb_list)

        assert len(trj_list) == len(
            candidate_methods), f"The number of the asses_list does not match the input methods."

        ''' Counts the number of feedback reasons for each methods. '''
        score_list = [[len(d) if d else 0 for d in tr] for tr in trj_list]

        assert len(score_list) == len(
            candidate_methods), f"The number of the score_list does not match the input methods."

        ''' Calculation of the average score. '''
        # scores = [np.mean(ls) for ls in score_list]  # average
        scores = [np.sqrt(np.mean(np.array(ls))) for ls in score_list]  # RMS
        assert len(scores) == len(candidate_methods), f"The number of the scores does not match the input methods."
        index = np.argmin(scores)
        # print('--scores---', scores)
        return mt_list[index], scores[index]

    def gen_feedbacks_from_seed_instructions(
            self,
            target_method: str,
            instructions_for_seed: Union[str, List[str]],
            num_instructions_to_evolve: int,
    ) -> List[str]:
        inst_list = [instructions_for_seed] if isinstance(instructions_for_seed, (str,)) else instructions_for_seed

        _, fb_list = self.evolve_multiple_instructions(
            instructions=inst_list,
            evol_method=target_method,
            evol_times=num_instructions_to_evolve,
        )

        feedbacks = self.integrate_multiple_feedbacks(
            fb_list,
            boundary_count=self._feedbacks_integrate_thresh,
        )

        return feedbacks

    def method_optimizer(
            self,
            target_method: str,
            instructions_for_seed: Union[str, List[str]],
            instructions_for_eval: Union[str, List[str]],
            num_instructions_to_evolve: int,
    ) -> Tuple[str, Any]:

        feedbacks = self.gen_feedbacks_from_seed_instructions(
            target_method=target_method,
            instructions_for_seed=instructions_for_seed,
            num_instructions_to_evolve=num_instructions_to_evolve,
        )

        candidate_methods = self.optimize_method_based_feedbacks(
            target_method=target_method,
            feedbacks=feedbacks,
        )

        opt_method, score = self.assessing_superior_methods(
            candidate_methods=candidate_methods,
            instructions_for_eval=instructions_for_eval,
            num_instructions_to_evolve=num_instructions_to_evolve,
        )

        return opt_method, score


if __name__ == "__main__":
    """
    python -m auto_evol.director
    """

    from model.groq_llm import GroqChatBase

    llm = GroqChatBase(
        model_name="llama-3.1-70b-versatile",
        requests_per_second=0.32,
        temperature=0.6,
    )


    aed = AutoEvolveDirector(
        main_llm=llm,
    )

    method_list = [
        'You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version. Please follow the steps below to rewrite the given "#Instruction#" into a more complex version. Step 1: Please read the "#Instruction#" carefully and list all the possible methods to make this instruction more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Please do not provide methods to change the language of the instruction! Additionally, provide a brief explanation for each method to justify its complexity increase. Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# more complex. The plan should include several methods from the #Methods List# and provide a detailed explanation of how these methods will increase the complexity of the instruction. Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only add 10 to 20 words into the "#Instruction#". Ensure that the added complexity is justified and explained in the plan. Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#. Just provide the #Finally Rewritten Instruction# along with a summary of the complexity increase in 1-2 sentences. Please reply strictly in the following format: Step 1 #Methods List#: Step 2 #Plan#: Step 3 #Rewritten Instruction#: Step 4 #Finally Rewritten Instruction#: ',
        # 'You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version while considering additional requirements to ensure the complexity increase is not lower than the previous method. Please follow the steps below to rewrite the given "#Instruction#" into a more complex version. Step 1: Please read the "#Instruction#" carefully, list all the possible methods to make this instruction more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle), and identify any potential additional requirements. Please do not provide methods to change the language of the instruction! Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# more complex, including methods to address the additional requirements. The plan should include several methods from the #Methods List#. Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only add 10 to 20 words into the "#Instruction#". Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#. Just provide the #Finally Rewritten Instruction# without any explanation. Please reply strictly in the following format: Step 1 #Methods List#: Step 2 #Plan#: Step 3 #Rewritten Instruction#: Step 4 #Finally Rewritten Instruction#: ',
        # 'You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version, ensuring that the complexity increase brought by the rewritten method is not lower than the previous method, and considering the provided context. Please follow the steps below to rewrite the given "#Instruction#" into a more complex version.\n\nStep 1: Please read the "#Instruction#" carefully, analyze the context, and list all the possible methods to make this instruction more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Please do not provide methods to change the language of the instruction!\n\nStep 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# more complex. The plan should include several methods from the #Methods List# and consider the context to avoid unreasonable complexity increases.\n\nStep 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only add 10 to 20 words into the "#Instruction#".\n\nStep 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#. Just provide the #Finally Rewritten Instruction# without any explanation.'
    ]

    test_inst = [
        '数学の問題を解くために、データを3つの数字のリストとして与えられます。集合論的な問題を解いてください。[3, 5, 1, 2, 1, 4, 5] の重複を削除した数字の和を求めてください。',
        '数学の問題の解法を説明して、答えを計算してください。服の店で10,000円分の購入を行い、20%の割引を受けました。買い物の合計が何円になりますか。',
        '与えられた数式を使って、問題を解決してください。太郎がお金を100円ずつ貯めていたが、遅れても大丈夫と考え、3日間は貯めることを忘れてしまいました。太郎は毎日100円を貯めています。太郎が週に貯めるお金の量を計算してください。',
    ]

    feedbacks = aed.gen_feedbacks_from_seed_instructions(
        target_method=method_list[0],
        instructions_for_seed=test_inst,
        num_instructions_to_evolve=10,
    )

    print(feedbacks)

