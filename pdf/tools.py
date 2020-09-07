import json
import logging
import multiprocessing as mp
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from logging import Logger, LogRecord, handlers
from typing import List, Dict, Any, TypeVar, Callable, Optional, TypedDict, Iterable, Tuple, Union, Generator, \
    NamedTuple

import numpy as np

from pdf.storage import Storage, SqlStorage

__all__ = [
    "get_files", "get_child_id", "get_text_dir", "get_data_dir", "get_indices_map_dir", "get_files_mapped",
    "get_child_total", "get_child_current", "get_config", "init_child_process", "do_in_parallel",
    "necessary_files", "get_storage", "set_storage", "ensure_list", "unique_words", "ResultSerializer",
    "LineResultSerializer", "file_results", "set_config", "process_files", "FileConfig", "update_config", "Stage",
    "get_extra", "File", "config_numbers"
]


class ConfigEntry(TypedDict):
    index: int
    last_modified: float


FileConfig = Dict[str, ConfigEntry]
T = TypeVar("T")
S = TypeVar("S")
Function = Callable[[mp.Pool, FileConfig, List[str]], Optional[T]]
ParallelFunction = Callable[[str, int], Optional[T]]
Mapper = Callable[[List[T], List[T]], S]

VALID_EXTENSIONS = ("pdf", "epub")


def get_files(directory: str, extensions: Iterable[str]) -> List[str]:
    files = []
    for dir_path, _, filenames in os.walk(directory):
        for name in filenames:
            valid_extension = False

            for extension in extensions:
                if name.endswith(extension):
                    valid_extension = True
                    break
            if valid_extension:
                file_path = os.path.join(dir_path, name)
                files.append(file_path)
    return files


Dir = Dict[str, List[str]]


def get_files_mapped(directory: str, extensions: Iterable[str]) -> Dir:
    dir_files: Dir = dict()

    for dir_path, _, filenames in os.walk(directory):
        dir_content = dir_files[dir_path] = []

        for name in filenames:
            valid_extension = False

            for extension in extensions:
                if name.endswith(extension):
                    valid_extension = True
                    break
            if valid_extension:
                file_path = os.path.join(dir_path, name)
                dir_content.append(file_path)
    return dir_files


child_current: Optional[mp.Value] = None
child_total: Optional[mp.Value] = None
child_id: Optional[mp.Value] = None


def get_child_current():
    return child_current


def get_child_total():
    return child_total


def get_child_id():
    return child_id


def init_child_process(current: mp.Value, total: mp.Value, id: mp.Value = None):
    global child_current
    global child_total
    global child_id
    child_current = current
    child_total = total
    child_id = id


class File(TypedDict):
    path: str
    result: Any
    x: Any
    td_if: Any
    words: Any
    index: Optional[int]


def unique_words(other: List[File], values: List[str]) -> List[str]:
    unique_values = set(values)
    for value in other:
        unique_values.update(value["words"])
    return list(unique_values)


STORAGE: Storage = SqlStorage()


def get_storage() -> Storage:
    return STORAGE


def set_storage(storage: Storage):
    global STORAGE
    STORAGE = storage


def get_text_dir() -> str:
    """
    Returns the path to the text dir.
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


def ensure_list(value):
    if isinstance(value, list):
        return value
    elif isinstance(value, np.ndarray):
        return value.tolist()
    else:
        return list(value)


def file_results(config: FileConfig, data_dir, ignore=None, serializer=None, fields=None, cached=None) \
        -> Generator[Dict[str, Any], None, None]:
    serializer = serializer or ResultSerializer()

    for file, file_config in config.items():
        index = file_config["index"]
        if ignore and index in ignore:
            continue

        if cached is not None and index in cached:
            yield cached[index]
            continue

        data_file = os.path.join(data_dir, f"{index}.json")

        if os.path.exists(data_file):
            fields = fields if cached is None else None
            result = serializer.deserialize_file(data_file, fields)

            if result["file"] != file:
                raise ValueError(f"Expected {file} but read {result['file']}")

            if cached is not None:
                cached[index] = result

            yield result
        else:
            pass
            # print(f"Expected File {data_file} to exist.")


def process_files(pool, config: FileConfig, files, function: ParallelFunction[T]) -> List[T]:
    start = datetime.now()

    update_config(config, files)

    # parallelize the function with the given pool
    results = pool.map(function, [(file, config[file]["index"]) for file in files])
    results = [result for result in results if result]
    print(f"Time taken: {datetime.now() - start}")
    return results


def config_numbers(config: FileConfig) -> List[int]:
    return [value["index"] for value in config.values()]


def update_config(config: "FileConfig", files: Union[List[str], Dir]) -> None:
    updated = False
    last_number = max(config_numbers(config), default=0)

    # ensure every file has an config_index
    for file in files:
        if file not in config:
            last_number += 1
            config[file] = to_config_entry(last_number, file)
            updated = True

    if updated:
        set_config(config)


def necessary_files(extensions: Iterable[str], files: Optional[List[str]], directory: str) \
        -> Tuple[Union[List[str], Dir], int]:
    extensions = extensions or VALID_EXTENSIONS

    if not files and not directory:
        raise ValueError("Neither files nor directory specified")
    if not files:
        files = get_files_mapped(directory, extensions)
        total_length = 0
        for value in files.values():
            total_length += len(value)
    else:
        total_length = len(files)

    return files, total_length


def do_in_parallel(function: Function[T], files: List[str], directory: str, extensions) -> List[T]:
    files, total_length = necessary_files(extensions, files, directory)
    config: FileConfig = get_config()

    total = mp.Value("i", total_length)
    current = mp.Value("i", 0)

    cpu_count = max(mp.cpu_count() - 1, 1)
    init_args = (current, total)
    with mp.Pool(cpu_count, initializer=init_child_process, initargs=init_args) as pool:
        if files.items:
            result = []
            for dir_path, content in files.items():
                intermediary_result = function(pool, config, content)
                result.extend(intermediary_result)
                print(f"{current.value}/{total.value}: Files Directory of '{dir_path}' finished.")
        else:
            result = function(pool, config, files)
    return result


class ResultSerializer:
    fields: List[str]

    def __init__(self):
        self.fields = ["file", "td_if", "words", "x"]

    def serialize(self, value: Dict[str, Any]) -> str:
        value = {key: value for key, value in value.items() if key in self.fields}
        return json.dumps(value)

    def deserialize(self, string: str) -> Dict[str, Any]:
        return {key: value for key, value in json.loads(string).items() if key in self.fields}

    def deserialize_file(self, path: str, fields=None) -> Dict[str, Any]:
        with open(path, "r", encoding="utf8") as file:
            return {key: value for key, value in json.load(file).items() if
                    key in self.fields and (not fields or key in fields)}


class LineResultSerializer(ResultSerializer):

    def __init__(self):
        super().__init__()
        self.split_pattern = re.compile(",\\s*")

    def serialize(self, value: Dict[str, Any]) -> str:
        result = ""
        for field in self.fields:
            field_value = value[field]

            if isinstance(field_value, (list, set)):
                # remove brackets
                result += str(field_value)[1:-1] + "\n"
            else:
                result += str(field_value) + "\n"

        return result

    def deserialize_file(self, path: str, fields=None) -> Dict[str, Any]:
        result = {}
        with open(path, "r", encoding="utf8") as file:
            max_field_index = len(self.fields) - 1
            for index, line in enumerate(file):

                if max_field_index < index:
                    break
                line = line.strip()
                field = self.fields[index]

                if fields and field not in fields:
                    continue

                if field == "file":
                    result[field] = line
                elif field == "td_if":
                    result[field] = np.fromstring(line, sep=",")
                elif field == "words":
                    result[field] = self.split_pattern.split(line)
                elif field == "x":
                    result[field] = np.fromstring(line, sep=",", dtype=int)
        return result

    def deserialize(self, string: str, fields=None) -> Dict[str, Any]:
        lines = string.splitlines()
        file = lines[0]

        td_if = np.fromstring(lines[1], sep=",")
        words = self.split_pattern.split(lines[2])
        x = np.fromstring(lines[3], sep=",", dtype=int)

        return {
            "file": file,
            "td_if": td_if,
            "words": words,
            "x": x
        }


def to_config_entry(config_index: int, file: str) -> ConfigEntry:
    return {"index": config_index, "last_modified": os.path.getmtime(file)}


def get_config() -> "FileConfig":
    data_dir = get_data_dir()
    path = os.path.join(data_dir, "config.json")

    if not os.path.exists(path):
        with open(path, "w"):
            return dict()
    else:
        with open(path, "r") as file:
            content = file.read()

        if not content.strip():
            return dict()

        config: "FileConfig" = json.loads(content)
        current_numbers = set(config_numbers(config))
        data_pattern = re.compile("^(\\d+)\\.json$")
        config_changed = False

        for file_name in os.listdir(data_dir):
            match = data_pattern.match(file_name)

            if not match:
                continue
            number = int(match.group(1))

            if number in current_numbers:
                continue

            with open(os.path.join(data_dir, file_name), "r") as file:
                content = json.load(file)
            original_file = content["file"]

            if original_file in config:
                print(f"{original_file} has result under number different than {config[original_file]}: {number}")
            else:
                config[original_file] = to_config_entry(number, original_file)
                config_changed = True

        if config_changed:
            set_config(config)
        return config


def set_config(config: "FileConfig"):
    data_dir = get_data_dir()

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    path = os.path.join(data_dir, "config.json")

    with open(path, "w") as file:
        json.dump(config, file)


def get_data_dir():
    path = os.path.join(os.getcwd(), "data")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_extra(**kwargs) -> Dict[str, Any]:
    task_pid = os.getpid()
    return {
        "date_time": datetime.now(),
        "pid": task_pid,
        "stage": None,
        "task_pid": task_pid,
        "total": 0,
        "current": 0,
        "state": None,
        **kwargs
    }


class ProcessRecord(LogRecord):
    reason: Optional[str]
    task_pid: int
    current: int
    total: int
    date_time: datetime
    pid: int
    stage: str
    state: Optional[str]


class RunConfig(NamedTuple):
    processes: int


class StageLogger(logging.Logger):

    def stage(self):
        pass


def create_logger(queue) -> StageLogger:
    logger = logging.getLogger("pdf")
    logger.setLevel(level=logging.INFO)

    stage_level = logging.INFO + 5

    def stage(self: StageLogger, msg, *args, **kwargs):
        if self.isEnabledFor(stage_level):
            self._log(stage_level, msg, args, **kwargs)

    logger.stage = stage.__get__(logger, None)

    if queue:
        logger.addHandler(handlers.QueueHandler(queue))
    # noinspection PyTypeChecker
    return logger


class Stage(ABC):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    FAILED = "failed"
    NONE = "none"

    """
    A Stage of Computation in this Module.
    Any Instances need to be completely pickleable (See pickling in multiprocessing.Pool).
    """

    def __init__(self, logger: Logger, name: str, run_config: RunConfig) -> None:
        self._logger = logger
        self._name = name
        self._task_pid = os.getpid()
        self._state = self.NONE
        self.run_config = run_config

    def compute_work(self) -> int:
        """
        Compute the total amount of work to do.
        Subclasses should cache that value and only recompute when necessary.
        """
        return 0

    def _get_logger(self):
        return self._logger

    def report_progress(self, value: Any, current=0, **kwargs):
        self._get_logger().info(
            value,
            extra=get_extra(
                state=self._state,
                current=current,
                total=self.compute_work(),
                stage=self._name,
                task_pid=self._task_pid,
                **kwargs
            )
        )

    def report_warning(self, value: Any, **kwargs):
        # ensure reason is a string
        if "reason" in kwargs:
            kwargs["reason"] = str(kwargs["reason"])

        self._get_logger().warning(
            value,
            extra=get_extra(
                state=self._state,
                stage=self._name,
                task_pid=self._task_pid,
                **kwargs
            )
        )

    def report_started(self):
        print(f"Started Stage: {self._name}")
        total = self.compute_work()
        self._state = self.RUNNING
        self._get_logger().info(
            f"Started Stage {self._name}",
            extra=get_extra(
                state=self._state,
                current=total,
                total=total,
                stage=self._name,
                task_pid=self._task_pid,
            )
        )

    def report_finished(self):
        print(f"Finished Stage: {self._name}")
        total = self.compute_work()
        self._state = self.SUCCEEDED
        self._get_logger().info(
            f"Finished Stage {self._name}",
            extra=get_extra(
                state=self._state,
                current=total,
                total=total,
                stage=self._name,
                task_pid=self._task_pid,
            )
        )

    def restore_logger(self, logger: Logger):
        if self._logger is None:
            self._logger = logger

    def recreate_logger(self, queue: Optional[mp.Queue]):
        if self._logger is None:
            self._logger = create_logger(queue)

    def remove_logger(self) -> Logger:
        logger = self._logger
        self._logger = None
        return logger

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def sequential(self, *args, **kwargs):
        pass

    def check(self, *args, **kwargs):
        pass

    def unchecked(self, *args, **kwargs):
        pass

    def parallel(self, *args, **kwargs):
        pass
