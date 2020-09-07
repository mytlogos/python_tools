import multiprocessing as mp
import re
from typing import Tuple, Optional

import pdfminer.high_level as miner
import textract

from pdf import tools

__all__ = ["ExtractStage"]

SYNC_CURRENT: Optional[mp.Value] = None
MESSAGE_QUEUE: Optional[mp.Queue] = None


def child_initializer(sync_current, queue):
    global SYNC_CURRENT
    global MESSAGE_QUEUE
    SYNC_CURRENT = sync_current
    MESSAGE_QUEUE = queue


class ExtractStage(tools.Stage):

    def __init__(self, logger, run_config) -> None:
        super().__init__(logger, "extract", run_config)
        self._total_work = None

    def compute_work(self) -> int:
        if self._total_work is None:
            self._total_work = tools.get_storage().get_text_model().compute_work()
        return self._total_work

    def run(self, files, config, queue):
        self.report_started()

        if self.run_config.processes <= 1:
            print("Running sequential")
            self.sequential(files, config)
        else:
            print("Running parallel")
            self.parallel(files, config, queue)
        self.report_finished()

    def sequential(self, files, config):
        self._total_work = len(files)

        for index, file in enumerate(files):
            file_config = config[file]
            self.check(file, file_config["index"])
            self.report_progress(f"Extracted from {file}", current=index + 1)

    def check(self, file: str, config_index: int, *, return_data=False) -> Optional[str]:
        model = tools.get_storage().get_text_model()

        if model.exists(config_index):
            if return_data:
                return model.get(config_index)
        else:
            value = self.unchecked(file, config_index, return_data=return_data)

            if return_data:
                return value

    def unchecked(self, file: str, config_index: int, *, return_data=False) -> Optional[str]:
        # extract the purely the text without any formatting
        try:
            if file.endswith(".pdf"):
                text = miner.extract_text(file)
            else:
                text = textract.process(file)
        except Exception as e:
            self.report_warning(f"Failed extracting from {file}", reason=e)
            return

        # ensure it is a string
        if not isinstance(text, str):
            text = str(text, "utf8")

        # unescape newlines
        pattern = re.compile("(\\+r)|(\\+n)|(\\+r\\+n)")
        sub = pattern.sub("\n", text).strip()

        tools.get_storage().get_text_model().save(config_index, sub)

        if not sub:
            self.report_warning("No data for " + file)

        if return_data:
            return sub

    def check_parallel(self, args: Tuple[str, int]):
        file, config_index = args
        self.recreate_logger(MESSAGE_QUEUE)

        self.check(file, config_index)

        with SYNC_CURRENT.get_lock():
            SYNC_CURRENT.value += 1
            current = SYNC_CURRENT.value

        self.report_progress(f"Extracted from {file}", current=current)

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
