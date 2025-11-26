from __future__ import annotations

import logging
import mmap
import re
from pathlib import Path
from typing import Iterator, TextIO

logger = logging.getLogger(__name__)


class ConstantinosKalapotharakosFormatError(Exception):
    pass


def constantinos_kalapotharakos_format_record_generator(path: Path, elements_per_record: int
                                                        ) -> Iterator[tuple[float, ...]]:
    """
    Create a record generator for a Constantinos Kalapotharakos format file.

    :param path: The path to the file.
    :param elements_per_record: The number of elements per record.
    :return: A generator that iterates over the records.
    """
    with path.open() as file_handle:
        file_contents = get_memory_mapped_file_contents(file_handle)
        generator = constantinos_kalapotharakos_format_record_generator_from_file_contents(
            file_contents=file_contents, elements_per_record=elements_per_record, file_path_for_logging=path)
        for record in generator:
            yield record


def constantinos_kalapotharakos_format_record_generator_from_file_contents(
        file_contents: bytes | mmap.mmap,
        *,
        elements_per_record: int,
        file_path_for_logging: Path | None = None,
) -> Iterator[tuple[float, ...]]:
    """
    Create a record generator for a Constantinos Kalapotharakos format file's contents.

    :param file_contents: The file contents object.
    :param elements_per_record: The number of elements per record.
    :return: A generator that iterates over the records.
    """
    value_iterator = re.finditer(rb"\S+", file_contents)
    count = 0
    values = []
    while True:
        try:
            value_string = next(value_iterator).group(0)
            try:
                values = [float(value_string)]
            except ValueError as error:
                if value_string.startswith(b'\x00'):
                    null_value_repeat_count = len(value_string) // len(b'\x00')
                    yields_to_repeat = null_value_repeat_count // 305
                    path_logging_string = f'{file_path_for_logging.name}: ' if file_path_for_logging is not None else ''
                    logger.warning(f'{path_logging_string}Encountered repeating `b\'\x00\'` string. Yielding previous record repeatedly.')
                    for _ in range(yields_to_repeat):
                        yield tuple(values)
                        count += 1
                    continue
                else:
                    raise error
        except StopIteration:
            break
        try:
            for _ in range(elements_per_record - 1):
                values.append(float(next(value_iterator).group(0)))
            yield tuple(values)
            if count % 100000 == 0:
                path_logging_string = f'{file_path_for_logging.name}: ' if file_path_for_logging is not None else ''
                logger.info(f'{path_logging_string}Processed {count} rows.')
            count += 1
        except StopIteration:
            raise ConstantinosKalapotharakosFormatError(
                f'The Constantinos Kalapotharakos format file ran out of elements when trying to get '
                f'{elements_per_record} elements for the current record.')


def get_memory_mapped_file_contents(file_handle: TextIO) -> mmap.mmap:
    """
    Get a memory mapped version of a file handle's contents.

    :param file_handle: The file handle to memory map.
    :return: The memory map.
    """
    file_fileno = file_handle.fileno()
    file_contents = mmap.mmap(file_fileno, 0, access=mmap.ACCESS_READ)
    return file_contents
