from abc import ABCMeta, abstractmethod

from typing import Dict, Optional, List
import re
import codecs

import json

class FileHandler(metaclass=ABCMeta):
    def __init__(self, encoding: str = "utf-8"):
        self._codec = encoding

    @staticmethod
    def suffix_check(suffix_list: List[str], file_name: str) -> bool:
        suffix_list = [s.strip(".") for s in suffix_list]
        suffix = "|".join(suffix_list).upper()
        pattern_str = r"\.(" + suffix + r")$"
        if not re.search(pattern_str, file_name.upper()):
            mes = f"""
            Suffix of the file name should be '{suffix}'.
            Current file name : {file_name}
            """
            print(mes)
            return False
        return True

    def read(self, file_name: str):
        try:
            with codecs.open(file_name, "r", self._codec) as srw:
                return self.read_handling(srw)
        except FileNotFoundError:
            print(f"{file_name} can not be found ...")
        except OSError as e:
            print(e)

    def write(self, data, file_name: str):
        try:
            with codecs.open(file_name, "w", self._codec) as srw:
                self.write_handling(data, srw)
        except OSError as e:
            print(e)

    @abstractmethod
    def read_handling(self, srw: codecs.StreamReaderWriter):
        raise NotImplementedError

    @abstractmethod
    def write_handling(self, data, srw: codecs.StreamReaderWriter):
        raise NotImplementedError


class JsonlHandler(FileHandler):
    def __init__(self, codec: str = "utf-8"):
        super().__init__(codec)

    def read_handling(self, srw: codecs.StreamReaderWriter) -> List[Dict]:
        jsonl_data = [json.loads(l) for l in srw.readlines()]
        return jsonl_data

    def write_handling(self, data: List[Dict], srw: codecs.StreamReaderWriter):
        data_cl = [json.dumps(d, ensure_ascii=False) + "\n" for d in data]
        srw.writelines(data_cl)


class JsonHandler(FileHandler):
    def __init__(self, codec: str = "utf-8"):
        super().__init__(codec)

    def read_handling(self, srw: codecs.StreamReaderWriter) -> Dict:
        return json.load(srw)

    def write_handling(self, data: Dict, srw: codecs.StreamReaderWriter):
        json.dump(data, srw, ensure_ascii=False, indent=2)


class JsonGenerationAnalyzer(FileHandler):
    def read_handling(self, srw: codecs.StreamReaderWriter):
        pass

    def write_handling(self, data, srw: codecs.StreamReaderWriter):
        pass


if __name__ == "__main__":
    """
    python -m util.file_tools
    """

    """ test for jsonl """

    file = "./data/test.jsonl"

    jlh = JsonlHandler()

    data = jlh.read(file)

    # print(type(data[0]))
    # print(data)

    dict_list = [
        {
            "id": "seed_task_0",
            "instruction": "朝食として、卵を使わず、タンパク質を含み、だいたい700～1000キロカロリーのものはありますか？",
        },
        {
            "id": "seed_task_1",
            "instruction": "与えられたペアの関係は？",
        },
        {
            "id": "seed_task_2",
            "instruction": "次の各人物について、それぞれ1文で説明しなさい。",
        },
    ]

    file = "./data/test_write.jsonl"
    jlh.write(dict_list, file_name=file)

    """ test for json """

    file = "./data/test.json"

    jh = JsonHandler()

    data = jh.read(file)

