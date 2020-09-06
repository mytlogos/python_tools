import re
from datetime import timedelta

import pdf

regex = re.compile(
    r"^((?P<days>[\\.\d]+?)d)?((?P<hours>[\\.\d]+?)h)?((?P<minutes>[\\.\d]+?)m)?((?P<seconds>[\\.\d]+?)s)?$"
)
TIME_DELTA_REGEX = re.compile(
    r"^((?P<hours>\d+):)?((?P<minutes>\d+):)?((?P<seconds>\d+)\.)?(?P<microseconds>\d+)$"
)


def parse_time(time_str):
    """
    Parse a time string e.g. (2h13m) into a timedelta object.

    Modified from virhilo's answer at https://stackoverflow.com/a/4628148/851699

    :param time_str: A string identifying a duration.  (eg. 2h13m)
    :return datetime.timedelta: A datetime.timedelta object
    """
    parts = regex.match(time_str)
    assert parts is not None, "Could not parse any time information from '{}'".format(time_str)
    time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
    return timedelta(**time_params)


def parse_time_delta(time_str):
    """
    Parse a time string e.g. (0:00:16.864681) into a timedelta object.

    Modified from virhilo's answer at https://stackoverflow.com/a/4628148/851699

    :param time_str: A string identifying a duration.  (eg. 0:00:16.864681)
    :return datetime.timedelta: A datetime.timedelta object
    """
    parts = TIME_DELTA_REGEX.match(time_str)
    assert parts is not None, "Could not parse any time information from '{}'".format(time_str)
    time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
    return timedelta(**time_params)


def status():
    success = []
    failed = []
    warnings = []
    success_time_taken_extract = timedelta()
    success_time_taken_total = timedelta()
    total_time_taken = timedelta()
    failed_pattern = re.compile("^Failed extracting from (.+): (.+)$")
    warning_pattern = re.compile("^PDFTextExtractionNotAllowedWarning: The PDF <_io.BufferedReader name='(.+)'>$")
    success_pattern = re.compile("^\\d+/\\d+ PID: (\\d+), Extracted from (.+), Time extracting: (.+), Total: (.+)$")
    single_run_pattern = re.compile("^Time taken: (.+)$")

    with open("output - Kopie.log", "r") as file:
        for line in file:
            match = failed_pattern.match(line)

            if match:
                path, message = match.group(1), match.group(2)
                failed.append((path, message))
                continue

            match = warning_pattern.match(line)

            if match:
                path = match.group(1)
                warnings.append(path)
                continue

            match = success_pattern.match(line)

            if match:
                path, extract, total = match.group(2), match.group(3), match.group(4)
                success_time_taken_extract += parse_time_delta(extract)
                success_time_taken_total += parse_time_delta(total)
                success.append((path, extract, total))
                continue

            match = single_run_pattern.match(line)

            if match:
                total = match.group(1)
                total_time_taken += parse_time_delta(total)
                continue

    newline = "\n"
    print(f"""
    Failed:\n{newline.join([": ".join(fail) for fail in failed])}
    Warnings: {len(warnings)}, Failed: {len(failed)}, Success: {len(success)}
    Extract Time For Success: {success_time_taken_extract}
    Misc Time For Success: {success_time_taken_total - success_time_taken_extract}
    Total Time For Success: {success_time_taken_total}
    Total Time for Failed: {total_time_taken - success_time_taken_total}
    Total Time: {total_time_taken}
    """)


def main():
    pdf.run(directory="D:\\Bücher\\")
    # pdf.plot()
    # status()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
