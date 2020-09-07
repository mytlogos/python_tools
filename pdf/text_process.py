import multiprocessing as mp
from datetime import datetime
from typing import Tuple, Optional

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from pdf.storage import Result
from pdf.tools import File, get_storage, ensure_list, Stage

__all__ = ["ProcessStage"]

SYNC_CURRENT: Optional[mp.Value] = None
MESSAGE_QUEUE: Optional[mp.Queue] = None


def child_initializer(sync_current, queue):
    global SYNC_CURRENT
    global MESSAGE_QUEUE
    SYNC_CURRENT = sync_current
    MESSAGE_QUEUE = queue


class ProcessStage(Stage):

    def __init__(self, logger, run_config) -> None:
        super().__init__(logger, "process", run_config)
        self._total_work = None

    def compute_work(self) -> int:
        if self._total_work is None:
            self._total_work = get_storage().get_result_model().compute_work()
        return self._total_work

    def run(self, files, config, queue):
        self.report_started()
        if self.run_config.processes <= 1:
            self.sequential(files, config)
        else:
            self.parallel(files, config, queue)
        self.report_finished()

    def sequential(self, files, config):
        self._total_work = len(files)

        for index, file in enumerate(files):
            file_config = config[file]
            try:
                self.check(file, file_config["index"])
                self.report_progress(f"Extracted from {file}", current=index + 1)
            except ValueError as e:
                self.report_warning(f"Failed Process File: {file}", reason=e)

    def check(self, file: str, config_index: int):
        result_model = get_storage().get_result_model()

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
        text_model = get_storage().get_text_model()

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

    def check_parallel(self, args: Tuple[str, int]):
        file, config_index = args
        self.recreate_logger(MESSAGE_QUEUE)

        try:
            self.check(file, config_index)
        except ValueError as e:
            self.report_warning(f"Failed Process File: {file}", reason=e)

        with SYNC_CURRENT.get_lock():
            SYNC_CURRENT.value += 1
            current = SYNC_CURRENT.value

        self.report_progress(f"Processed {file}", current=current)

    def parallel(self, files, config, queue):
        self._total_work = len(files)
        current = mp.Value("i", 0)

        # remove logger cleanly as it will not be pickled 'correctly' (with config and handlers retained)
        logger = self.remove_logger()

        try:
            with mp.Pool(self.run_config.processes, initializer=child_initializer, initargs=(current, queue)) as pool:
                pool.map(self.check_parallel, [(file, config[file]["index"]) for file in files])
        finally:
            # restore logger, even if pool execution fails
            self.restore_logger(logger)
