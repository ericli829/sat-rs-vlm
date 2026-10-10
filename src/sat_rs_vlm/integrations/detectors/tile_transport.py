"""Lossless, inexpensive file transport for synchronous detector sidecars."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from PIL import Image

from .protocol import ProposalError

_Input = TypeVar("_Input")
_Prepared = TypeVar("_Prepared")


@contextmanager
def cpu_prepared_items(
    items: Iterable[_Input],
    prepare: Callable[[_Input], _Prepared],
    release: Callable[[_Prepared], None],
    *,
    prefetch: bool = True,
) -> Iterator[Iterator[_Prepared]]:
    """Prepare at most one item ahead; inference stays on the consuming thread.

    Closing the context joins CPU preparation and releases unused results even
    when inference fails or the caller stops consuming. No CUDA work belongs in
    ``prepare``. There are at most two prepared items: active plus lookahead.
    """

    source = iter(items)
    pending: Future[list[_Prepared]] | None = None
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-cpu") if prefetch else None

    def prepare_next() -> list[_Prepared]:
        try:
            item = next(source)
        except StopIteration:
            return []
        # Lazy image/crop producers run on the CPU thread, never the inference thread.
        return [prepare(item)]

    def iterate() -> Generator[_Prepared, None, None]:
        nonlocal pending
        if pool is None:
            for item in source:
                prepared = prepare(item)
                try:
                    yield prepared
                finally:
                    release(prepared)
            return
        pending = pool.submit(prepare_next)
        while pending is not None:
            completed = pending.result()
            pending = None
            if not completed:
                break
            prepared = completed[0]
            try:
                pending = pool.submit(prepare_next)
                yield prepared
            finally:
                release(prepared)

    iterator = iterate()
    try:
        yield iterator
    finally:
        try:
            iterator.close()
        finally:
            try:
                if pool is not None:
                    if pending is not None:
                        pending.cancel()
                    pool.shutdown(wait=True, cancel_futures=True)
                    if pending is not None and not pending.cancelled():
                        # A preparation failure has no completed resource to release.
                        if pending.exception() is None:
                            for prepared in pending.result():
                                release(prepared)
            finally:
                close = getattr(source, "close", None)
                if callable(close):
                    close()


def tile_image_format(value: str) -> str:
    image_format = str(value).strip().lower()
    if image_format not in {"bmp", "png"}:
        raise ProposalError("tile_image_format must be 'bmp' or 'png'")
    return image_format


@contextmanager
def cpu_prepared_batches(
    items: Iterable[_Input],
    prepare: Callable[[_Input], _Prepared],
    release: Callable[[_Prepared], None],
    *,
    batch_size: int,
    prefetch: bool = True,
) -> Iterator[Iterator[list[_Prepared]]]:
    """Keep at most an active batch and one CPU lookahead batch.

    Prepare each item before advancing its source: counting generators close
    the previous crop on next(), so collecting raw requests first is unsafe.
    """
    if batch_size < 1:
        raise ProposalError("detector batch_size must be positive")
    source = iter(items)

    def release_batch(batch: list[_Prepared]) -> None:
        for item in batch:
            release(item)

    def batches() -> Iterator[list[_Prepared]]:
        while True:
            batch: list[_Prepared] = []
            try:
                for _ in range(batch_size):
                    item = next(source, None)
                    if item is None:
                        break
                    batch.append(prepare(item))
            except BaseException:
                release_batch(batch)
                raise
            if not batch:
                return
            yield batch

    try:
        with cpu_prepared_items(
            batches(), lambda batch: batch, release_batch, prefetch=prefetch
        ) as prepared:
            yield prepared
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()


def save_tile_image(image: Image.Image, path: Path, image_format: str) -> None:
    """BMP avoids compression work; PNG remains available for constrained storage."""

    if image.mode == "RGB":
        image.save(path, format=image_format.upper())
    else:
        converted = image.convert("RGB")
        try:
            converted.save(path, format=image_format.upper())
        finally:
            converted.close()
