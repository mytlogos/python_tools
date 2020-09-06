import json
import os
import re
import sqlite3 as sql
from abc import abstractmethod, ABC
from dataclasses import dataclass
from sqlite3.dbapi2 import Connection
from typing import Dict, Any, List, Tuple, Iterable, Optional, Union

import numpy as np

__all__ = ["FileStorage", "SqlStorage", "Storage", "Result", "FileIndices", "TextModel", "ResultModel",
           "SimilarityModel", "CommonIndicesModel", "MapIndicesModel"]

FileIndices = Dict[int, int]


def numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    else:
        return np.asarray(value)


@dataclass
class Result:
    index: int
    file: str
    words: Union[List[str], Dict[int, int]]
    td_if: np.ndarray
    x: np.ndarray


class Storage(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_text_model(self) -> "TextModel":
        pass

    @abstractmethod
    def get_result_model(self) -> "ResultModel":
        pass

    @abstractmethod
    def get_map_indices_model(self) -> "MapIndicesModel":
        pass

    @abstractmethod
    def get_similarity_model(self) -> "SimilarityModel":
        pass


class TextModel(ABC):
    @abstractmethod
    def compute_work(self) -> int:
        pass

    @abstractmethod
    def save(self, index: int, text: str) -> None:
        pass

    @abstractmethod
    def save_many(self, param: List[Tuple[int, str]]) -> None:
        pass

    @abstractmethod
    def exists(self, index: int) -> bool:
        pass

    @abstractmethod
    def get(self, index: int) -> str:
        pass

    @abstractmethod
    def get_many(self, index: Iterable[int]) -> List[str]:
        pass


class ResultModel(ABC):
    @abstractmethod
    def compute_work(self) -> int:
        pass

    @abstractmethod
    def save(self, result: Result) -> None:
        pass

    @abstractmethod
    def save_many(self, param: List[Result]) -> None:
        pass

    @abstractmethod
    def exists(self, index: int) -> bool:
        pass

    @abstractmethod
    def get(self, index: int) -> Optional[Result]:
        pass

    @abstractmethod
    def get_many(self, index: Iterable[int]) -> List[Result]:
        pass


class MapIndicesModel(ABC):
    @abstractmethod
    def compute_work(self) -> int:
        pass

    @abstractmethod
    def save(self, index: int, text: Union[str, Dict[int, int]]) -> None:
        pass

    @abstractmethod
    def save_many(self, param: List[Tuple[int, Union[str, Dict[int, int]]]]) -> None:
        pass

    @abstractmethod
    def exists(self, index: int) -> bool:
        pass

    @abstractmethod
    def get(self, index: int) -> Optional[Dict[int, int]]:
        pass

    @abstractmethod
    def get_many(self, index: Iterable[int]) -> List[Dict[int, int]]:
        pass


class CommonIndicesModel(ABC):
    @abstractmethod
    def compute_work(self) -> int:
        pass

    @abstractmethod
    def save(self, index: int, other_index: int, common: np.ndarray, other_common: np.ndarray) -> None:
        pass

    @abstractmethod
    def save_many(self, param: List[Tuple[int, int, np.ndarray, np.ndarray]]) -> None:
        pass

    @abstractmethod
    def exists(self, index: int, other_index: int) -> bool:
        pass

    @abstractmethod
    def get(self, index: int, other_index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pass

    @abstractmethod
    def get_many(self, param: List[Tuple[int, int]]) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
        pass


class SimilarityModel(ABC):
    @abstractmethod
    def compute_work(self) -> int:
        pass

    @abstractmethod
    def save(self, param: Tuple[int, int, float]) -> None:
        pass

    @abstractmethod
    def save_many(self, param: Iterable[Tuple[int, int, float]]) -> None:
        pass

    @abstractmethod
    def exists(self, index_pair: Tuple[int, int]) -> bool:
        pass

    @abstractmethod
    def get(self, index_pair: Tuple[int, int]) -> Optional[float]:
        pass

    @abstractmethod
    def get_many(self, param: Iterable[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        pass


def get_data_dir():
    return os.path.join(os.getcwd(), "data")


def get_text_dir() -> str:
    """
    Returns the path to the text directory.
    Creates it if it does not exist.

    :rtype: str
    """
    path = os.path.join(get_data_dir(), "text")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_indices_map_dir() -> str:
    """
        Returns the path to the indices dir.
        Creates it if it does not exist.

        :rtype: str
        """
    path = os.path.join(get_data_dir(), "indices")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


class FileHelper:
    def __init__(self):
        self._encoding = "utf8"
        self.data_dir = get_data_dir()
        self.text_dir = get_text_dir()
        self.indices_dir = get_indices_map_dir()

    @staticmethod
    def exists(directory: str, index: int, extension="json") -> bool:
        path = os.path.join(directory, f"{index}.{extension}")
        return os.path.exists(path)

    def save_json(self, directory, index, value) -> None:
        path = os.path.join(directory, f"{index}.json")

        with open(path, "w", encoding=self._encoding) as text_file:
            json.dump(value, text_file)

    def load_json(self, directory, index) -> Any:
        text = self.load(directory, f"{index}.json")
        return json.loads(text)

    def save(self, directory: str, file: str, text: str) -> None:
        path = os.path.join(directory, file)

        with open(path, "w", encoding=self._encoding) as text_file:
            text_file.write(text)

    def load(self, directory: str, file: str) -> Any:
        path = os.path.join(directory, file)

        with open(path, "r", encoding=self._encoding) as text_file:
            return text_file.read()


def count(directory: str, pattern: re.Pattern):
    file_count = 0
    for name in os.listdir(directory):
        if pattern.match(name):
            file_count += 1
    return file_count


class FileStorage(Storage):
    def __init__(self):
        super().__init__()
        helper = FileHelper()
        self.txt_model = FileTextModel(helper)
        self.result_model = FileResultModel(helper)
        self.common_indices_model = FileCommonIndicesModel(helper)
        self.map_indices_model = FileMapIndicesModel(helper)
        self.similarity_model = FileSimilarityModel(helper)

    def get_text_model(self) -> "TextModel":
        return self.txt_model

    def get_result_model(self) -> "ResultModel":
        return self.result_model

    def get_map_indices_model(self) -> "MapIndicesModel":
        return self.map_indices_model

    def get_similarity_model(self) -> "SimilarityModel":
        return self.similarity_model


class FileTextModel(TextModel):
    def __init__(self, helper: FileHelper):
        self.helper = helper

    def compute_work(self) -> int:
        return count(self.helper.text_dir, re.compile("\\d+\\.json"))

    def save(self, index: int, text: str) -> None:
        return self.helper.save(self.helper.text_dir, f"{index}.txt", text)

    def save_many(self, param: List[Tuple[int, str]]) -> None:
        for data in param:
            self.save(*data)

    def exists(self, index: int) -> bool:
        return self.helper.exists(self.helper.text_dir, index, extension="txt")

    def get(self, index: int) -> str:
        return self.helper.load(self.helper.text_dir, f"{index}.txt")

    def get_many(self, indices: Iterable[int]) -> List[str]:
        return [self.get(index) for index in indices]


class FileResultModel(ResultModel):
    def __init__(self, helper: FileHelper):
        self.helper = helper

    def compute_work(self) -> int:
        return count(self.helper.data_dir, re.compile("\\d+\\.json"))

    def save(self, result: Result) -> None:
        return self.helper.save_json(self.helper.data_dir, result.index, result)

    def save_many(self, param: List[Result]) -> None:
        for data in param:
            self.save(data)

    def exists(self, index: int) -> bool:
        return self.helper.exists(self.helper.data_dir, index)

    def get(self, index: int) -> Optional[Result]:
        return Result(index=index, **self.helper.load_json(self.helper.data_dir, index))

    def get_many(self, indices: Iterable[int]) -> List[Result]:
        return [self.get(index) for index in indices]


class FileMapIndicesModel(MapIndicesModel):
    def __init__(self, helper: FileHelper):
        self.helper = helper

    def compute_work(self) -> int:
        return count(self.helper.indices_dir, re.compile("\\d+\\.json"))

    def save(self, index: int, text: Union[str, Dict[int, int]]) -> None:
        return self.helper.save_json(self.helper.indices_dir, index, text)

    def save_many(self, param: List[Tuple[int, Union[str, Dict[int, int]]]]) -> None:
        for data in param:
            self.save(*data)

    def exists(self, index: int) -> bool:
        return self.helper.exists(self.helper.indices_dir, index)

    def get(self, index: int) -> Optional[Dict[int, int]]:
        return self.helper.load_json(self.helper.indices_dir, index)

    def get_many(self, indices: Iterable[int]) -> List[Dict[int, int]]:
        return [self.get(index) for index in indices]


class FileCommonIndicesModel(CommonIndicesModel):
    def __init__(self, helper: FileHelper):
        self.helper = helper

    def compute_work(self) -> int:
        return 0

    def save(self, index: int, other_index: int, common: np.ndarray, other_common: np.ndarray) -> None:
        raise NotImplemented()

    def save_many(self, param: List[Tuple[int, int, np.ndarray, np.ndarray]]) -> None:
        raise NotImplemented()

    def exists(self, index: int, other_index: int) -> bool:
        raise NotImplemented()

    def get(self, index: int, other_index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplemented()

    def get_many(self, param: List[Tuple[int, int]]) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
        raise NotImplemented()


class FileSimilarityModel(SimilarityModel):
    def __init__(self, helper: FileHelper):
        self.helper = helper

    def compute_work(self) -> int:
        result_counts = count(self.helper.data_dir, re.compile("\\d+\\.json"))
        squared = result_counts * result_counts
        # diagonal is not calculated and of difference only half needs to be calculated as the matrix is symmetric
        return (squared - result_counts) // 2

    def save(self, param: Tuple[int, int, float]) -> None:
        raise NotImplemented()

    def save_many(self, param: Iterable[Tuple[int, int, float]]) -> None:
        raise NotImplemented()

    def exists(self, index_pair: Tuple[int, int]) -> bool:
        raise NotImplemented()

    def get(self, index_pair: Tuple[int, int]) -> Optional[float]:
        raise NotImplemented()

    def get_many(self, param: Iterable[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        raise NotImplemented()


class SqlStorage(Storage):
    con: Connection

    def __init__(self):
        super().__init__()
        self.con = sql.connect("data.db")
        self.text_model = SqlTextModel(self.con)
        self.result_model = SqlResultModel(self.con)
        self.map_indices_model = SqlMapIndicesModel(self.con)
        self.common_indices_model = SqlCommonIndicesModel(self.con)
        self.similarity_model = SqlSimilarityModel(self.con)
        self._init()

    def get_text_model(self) -> "TextModel":
        return self.text_model

    def get_result_model(self) -> "ResultModel":
        return self.result_model

    def get_map_indices_model(self) -> "MapIndicesModel":
        return self.map_indices_model

    def get_similarity_model(self) -> "SimilarityModel":
        return self.similarity_model

    def _init(self):
        cursor = self.con.execute("SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%'")
        table_names = cursor.fetchall()

        if self.text_model.name not in table_names:
            self.text_model.create_table()

        if self.result_model.name not in table_names:
            self.result_model.create_table()

        if self.map_indices_model.name not in table_names:
            self.map_indices_model.create_table()

        if self.common_indices_model.name not in table_names:
            self.common_indices_model.create_table()

        if self.similarity_model.name not in table_names:
            self.similarity_model.create_table()

        self.con.commit()

    def close(self):
        self.con.close()


class SqlTextModel(TextModel):

    def __init__(self, con: Connection):
        self.name = "text"
        self.con = con

    def compute_work(self) -> int:
        cursor = self.con.execute("SELECT COUNT(id) FROM 'text'")
        row = cursor.fetchone()
        return row[0]

    def create_table(self) -> None:
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS 'text' (id INT PRIMARY KEY, 'text' TEXT NOT NULL)"
        )

    def save(self, index: int, text: str) -> None:
        self.con.execute("INSERT OR REPLACE into 'text' VALUES (?, ?)", (index, text))
        self.con.commit()

    def save_many(self, param: List[Tuple[int, str]]) -> None:
        self.con.executemany("INSERT OR REPLACE into 'text' VALUES (?, ?)", param)
        self.con.commit()

    def exists(self, index: int) -> bool:
        cursor = self.con.execute("SELECT 1 FROM 'text' WHERE id = ? LIMIT 1", [index])
        return True if cursor.fetchone() else False

    def get(self, index: int) -> Optional[str]:
        cursor = self.con.execute("SELECT 'text' FROM 'text' WHERE id = ?", [index])
        row = cursor.fetchone()
        return row[0] if row else None

    def get_many(self, indices: Iterable[int]) -> List[Tuple[int, str]]:
        result = []
        for index in indices:
            datum = self.get(index)
            if datum:
                result.append((index, datum))
        return result


class SqlResultModel(ResultModel):

    def __init__(self, con: Connection):
        self.name = "results"
        self.con = con

    def compute_work(self) -> int:
        cursor = self.con.execute("SELECT COUNT(id) FROM 'results'")
        row = cursor.fetchone()
        return row[0]

    def create_table(self) -> None:
        self.con.execute(
            f"CREATE TABLE IF NOT EXISTS 'results' "
            "('id' INT PRIMARY KEY, file TEXT NOT NULL, td_if BLOB NOT NULL, x BLOB NOT NULL, words TEXT NOT NULL)"
        )

    def save(self, result: Result) -> None:
        self.con.execute(f"INSERT OR REPLACE into 'results' VALUES (?, ?, ?, ?, ?)", self._result_to_tuple(result))
        self.con.commit()

    def save_many(self, param: List[Result]) -> None:
        inserts = [self._result_to_tuple(row) for row in param]
        self.con.executemany(f"INSERT OR REPLACE into 'results' VALUES (?, ?, ?, ?, ?)", inserts)
        self.con.commit()

    def exists(self, index: int) -> bool:
        cursor = self.con.execute("SELECT 1 FROM results WHERE id = ? LIMIT 1", [index])
        return True if cursor.fetchone() else False

    def get(self, index: int) -> Optional[Result]:
        cursor = self.con.execute("SELECT id, file, td_if, x, words FROM results WHERE id = ? LIMIT 1", [index])
        fetchone = cursor.fetchone()
        return self._tuple_to_result(fetchone) if fetchone else None

    def get_many(self, indices: Iterable[int]) -> List[Result]:
        # TODO: one cannot use cursor.executemany to query values (select * ) (doc says nothing)
        result = []
        for index in indices:
            datum = self.get(index)
            if datum:
                result.append(datum)
        return result

    @staticmethod
    def _result_to_tuple(result: Result) -> Tuple[int, str, bytes, bytes, str]:
        return result.index, result.file, numpy_array(result.td_if).tobytes(), \
               numpy_array(result.x).tobytes(), json.dumps(result.words)

    @staticmethod
    def _tuple_to_result(row: Tuple[int, str, bytes, bytes]) -> Result:
        return Result(
            index=row[0],
            file=row[1],
            td_if=np.frombuffer(row[2], dtype=float),
            x=np.frombuffer(row[3], dtype=int),
            words=json.loads(row[4])
        )


class SqlMapIndicesModel(MapIndicesModel):

    def __init__(self, con: Connection):
        self.name = "map_indices"
        self.con = con

    def compute_work(self) -> int:
        cursor = self.con.execute("SELECT COUNT(id) FROM map_indices")
        row = cursor.fetchone()
        return row[0]

    def create_table(self) -> None:
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS map_indices (id INT PRIMARY KEY, mapping TEXT NOT NULL)"
        )

    def save(self, index: int, text: Union[str, Dict[int, int]]) -> None:
        if not isinstance(text, str):
            text = json.dumps(text)

        self.con.execute("INSERT OR REPLACE into map_indices VALUES (?, ?)", (index, text))
        self.con.commit()

    def save_many(self, param: List[Tuple[int, Union[str, Dict[int, int]]]]) -> None:
        for index, value in enumerate(param):
            if not isinstance(value[1], str):
                text = json.dumps(value[1])
                param[index] = value[0], text

        self.con.executemany("INSERT OR REPLACE into map_indices VALUES (?, ?)", param)
        self.con.commit()

    def exists(self, index: int) -> bool:
        cursor = self.con.execute("SELECT 1 FROM map_indices WHERE id = ? LIMIT 1", [index])
        return True if cursor.fetchone() else False

    def get(self, index: int) -> Optional[Dict[int, int]]:
        cursor = self.con.execute("SELECT mapping FROM map_indices WHERE id = ? LIMIT 1", [index])
        fetchone = cursor.fetchone()
        return json.loads(fetchone[0]) if fetchone else None

    def get_many(self, indices: Iterable[int]) -> List[Dict[int, int]]:
        result = []
        for index in indices:
            datum = self.get(index)
            if datum:
                result.append(datum)
        return result


class SqlCommonIndicesModel(CommonIndicesModel):

    def __init__(self, con: Connection):
        self.name = "common_indices"
        self.con = con

    def compute_work(self) -> int:
        cursor = self.con.execute("SELECT COUNT(id) FROM common_indices")
        row = cursor.fetchone()
        return row[0]

    def create_table(self) -> None:
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS common_indices "
            "(id INT NOT NULL, other_id INT NOT NULL, "
            "common_indices BLOB NOT NULL, other_common_indices BLOB NOT NULL, "
            "PRIMARY KEY(id, other_id))"
        )

    def save(self, index: int, other_index: int, common: np.ndarray, other_common: np.ndarray) -> None:
        self.con.execute(
            "INSERT OR REPLACE into common_indices VALUES (?, ?, ?, ?)",
            (index, other_index, common.tobytes(), other_common.tobytes())
        )
        self.con.commit()

    def save_many(self, param: List[Tuple[int, int, np.ndarray, np.ndarray]]) -> None:
        param = [(row[0], row[1], row[2].tobytes(), row[3].tobytes()) for row in param]

        self.con.executemany(f"INSERT OR REPLACE into common_indices VALUES (?, ?, ?, ?)", param)
        self.con.commit()

    def exists(self, index: int, other_index: int) -> bool:
        cursor = self.con.execute(
            "SELECT 1 FROM common_indices WHERE id = ? AND other_id = ? LIMIT 1",
            (index, other_index)
        )
        return True if cursor.fetchone() else False

    def get(self, index: int, other_index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        cursor = self.con.execute(
            "SELECT common_indices, other_common_indices FROM common_indices WHERE id = ? AND other_id = ? LIMIT 1",
            (index, other_index)
        )
        fetchone = cursor.fetchone()

        if fetchone:
            return np.frombuffer(fetchone[0], dtype=int), np.frombuffer(fetchone[1], dtype=int)

    def get_many(self, param: List[Tuple[int, int]]) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
        result = []
        for datum in param:
            cursor = self.con.execute(
                "SELECT id, other_id, common_indices, other_common_indices "
                "FROM common_indices WHERE id = ? AND other_id = ?",
                datum
            )
            row = cursor.fetchone()
            if row:
                result.append((row[0], row[1], np.frombuffer(row[2], dtype=int), np.frombuffer(row[3], dtype=int)))
        return result


class SqlSimilarityModel(SimilarityModel):

    def __init__(self, con: Connection):
        self.name = "similarity"
        self.con = con

    def compute_work(self) -> int:
        cursor = self.con.execute("SELECT COUNT(*) FROM similarity")
        row = cursor.fetchone()
        return row[0]

    def create_table(self) -> None:
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS similarity "
            "(id INT NOT NULL, other_id INT NOT NULL, "
            "similarity REAL NOT NULL, "
            "PRIMARY KEY(id, other_id))"
        )

    def save(self, param: Tuple[int, int, float]) -> None:
        self.con.execute(
            "INSERT OR REPLACE into similarity VALUES (?, ?, ?)",
            param
        )
        self.con.commit()

    def save_many(self, param: Iterable[Tuple[int, int, float]]) -> None:
        self.con.executemany(f"INSERT OR REPLACE into similarity VALUES (?, ?, ?)", param)
        self.con.commit()

    def exists(self, index_pair: Tuple[int, int]) -> bool:
        cursor = self.con.execute(
            "SELECT 1 FROM similarity WHERE id = ? AND other_id = ? LIMIT 1",
            index_pair
        )
        return True if cursor.fetchone() else False

    def get(self, index_pair: Tuple[int, int]) -> Optional[float]:
        cursor = self.con.execute(
            "SELECT similarity FROM similarity WHERE id = ? AND other_id = ? LIMIT 1",
            index_pair
        )
        fetchone = cursor.fetchone()

        if fetchone:
            return fetchone[0]

    def get_many(self, param: Iterable[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        result = []
        for data in param:
            cursor = self.con.execute(
                "SELECT id, other_id, similarity "
                "FROM similarity WHERE id = ? AND other_id = ?",
                data
            )
            row = cursor.fetchone()

            if row:
                result.append((row[0], row[1], row[2]))
        return result
