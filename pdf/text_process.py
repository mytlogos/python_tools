import os
from datetime import datetime
from typing import Tuple, Optional, List

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from pdf.storage import Result, TextModel, ResultModel
from pdf.tools import process_files, File, STORAGE, ensure_list, get_child_current, get_child_total, Stage

__all__ = ["ProcessStage"]


class ProcessStage(Stage):

    def __init__(self, logger, text_model: TextModel, result_model: ResultModel) -> None:
        super().__init__(logger, "process")
        self._text_model = text_model
        self._result_model = result_model
        self._total_work = None

    def compute_work(self) -> int:
        if self._total_work is None:
            self._total_work = self._result_model.compute_work()
        return self._total_work

    def sequential(self, files, config):
        self._total_work = len(files)
        self.report_started()

        for index, file in enumerate(files):
            file_config = config[file]
            try:
                self.check((file, file_config["index"]))
                self.report_progress(f"Extracted from {file}", current=index + 1)
            except ValueError as e:
                self.report_warning(f"Failed Process File: {file}", reason=e)

        self.report_finished()

    def check(self, args: Tuple[str, int], **kwargs):
        file, config_index = args

        result_model = STORAGE.get_result_model()

        if result_model.exists(config_index):
            result = result_model.get(config_index)

            if result.file != file:
                raise ValueError(f"Expected {file} but read {result.file}")

            file_result = File(path=file, result=None, x=result.x, td_if=result.td_if, words=result.words,
                               index=config_index)
        else:
            file_result, extracted = self.unchecked(config_index, file, result_model)
        return file_result

    def unchecked(self, config_index, file, result_model, *args, **kwargs):
        text_model = STORAGE.get_text_model()

        if text_model.exists(config_index):
            text_data = text_model.get(config_index).strip()
        else:
            raise ValueError(f"Text Extract Stage not run for {config_index}")

        if not text_data:
            return

        extracted = datetime.now()

        vectorizer = CountVectorizer(stop_words='english')
        try:
            x = vectorizer.fit_transform([text_data]).toarray()
        except ValueError as e:
            self.report_progress(f"Failed for '{config_index}'", reason=e)
            return

        td_if = TfidfTransformer().fit_transform(x)
        x, = x
        td_if, = td_if.toarray()
        words = vectorizer.get_feature_names()

        value = Result(index=config_index, file=file, words=ensure_list(words), td_if=td_if, x=x)
        result_model.save(value)
        return File(path=file, result=None, x=x, td_if=td_if, words=words, index=config_index), extracted


def get_file_results(pool, config, files) -> List[File]:
    return process_files(pool, config, files, get_file_result)


def get_file_result(args: Tuple[str, int]) -> Optional[File]:
    child_current = get_child_current()
    child_total = get_child_total()

    file, config_index = args
    start = datetime.now()

    result_model = STORAGE.get_result_model()

    if result_model.exists(config_index):
        result = result_model.get(config_index)

        if result.file != file:
            raise ValueError(f"Expected {file} but read {result.file}")

        td_if = result.td_if
        X = result.x
        words = result.words
        extracted = datetime.now()
    else:
        text_model = STORAGE.get_text_model()

        if text_model.exists(config_index):
            text_data = text_model.get(config_index).strip()
        else:
            raise ValueError(f"Text Extract Stage not run for {config_index}")

        if not text_data:
            return

        extracted = datetime.now()

        vectorizer = CountVectorizer(stop_words='english')
        try:
            X = vectorizer.fit_transform([text_data]).toarray()
        except ValueError as e:
            print(f"Failed for '{file}': {e}")
            return
        td_if = TfidfTransformer().fit_transform(X)
        X = X[0]
        td_if = td_if.toarray()[0]
        words = vectorizer.get_feature_names()

        value = Result(index=config_index, file=file, words=ensure_list(words), td_if=td_if, x=X)
        result_model.save(value)

    end = datetime.now()

    with child_current.get_lock():
        child_current.value += 1
        extract_duration = extracted - start
        total = end - start
        print(f"{child_current.value}/{child_total.value} PID: {os.getpid()}, Extracted from {file},"
              f" Time extracting: {extract_duration}, Total: {total}")

    return File(path=file, result=None, x=X, td_if=td_if, words=words, index=config_index)
