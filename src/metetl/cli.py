"""CLI metetl: prepare / process / analyze."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from metetl import __version__
from metetl.logging_config import configure, get_logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metetl",
        description="MET ETL: подготовка списка картин, скачивание+обработка, анализ.",
    )
    parser.add_argument("--version", action="version", version=f"metetl {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="консоль на уровне DEBUG (по умолчанию INFO)")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="путь к файлу лога (по умолчанию logs/metetl.log)")

    sub = parser.add_subparsers(dest="command", required=True, metavar="команда")

    p_prep = sub.add_parser("prepare", help="отобрать картины из CSV и сохранить в JSON")
    p_prep.add_argument("--csv", type=Path, required=True, help="путь к MetObjects.csv")
    p_prep.add_argument("--output", type=Path, required=True,
                        help="куда положить to_download.json")
    p_prep.add_argument("--limit", type=int, default=None,
                        help="ограничить количество картин (по умолчанию: все)")
    p_prep.add_argument("--seed", type=int, default=None, help="seed для shuffle")

    p_proc = sub.add_parser("process", help="скачать N картин и применить свёртку")
    p_proc.add_argument("--input", type=Path, required=True, help="to_download.json")
    p_proc.add_argument("--output", type=Path, default=Path("images"),
                        help="директория, в которой создаётся run_<ts>/")
    p_proc.add_argument("--num", type=int, required=True, help="сколько картин обработать")
    p_proc.add_argument("--impl", choices=("numpy", "opencv", "c"), default="numpy",
                        help="реализация свёртки в воркере")
    p_proc.add_argument("--streaming", action="store_true",
                        help="использовать пайплайн на async-генераторах")

    p_an = sub.add_parser("analyze", help="анализ датасета (Lab 3)")
    p_an.add_argument("--csv", type=Path, required=True, help="путь к MetObjects.csv")
    p_an.add_argument("--output-dir", type=Path, required=True,
                       help="директория для сохранения графиков")

    return parser


def _cmd_prepare(args: argparse.Namespace, log) -> int:
    from metetl.analysis.data_to_download import prepare_to_download

    log.info("Команда prepare: csv=%s, output=%s, limit=%s",
             args.csv, args.output, args.limit)
    written = prepare_to_download(args.csv, args.output,
                                    limit=args.limit, seed=args.seed)
    if written == 0:
        log.error("Ничего не записано")
        return 1
    log.info("Готово: %d картин в %s", written, args.output)
    return 0


def _cmd_process(args: argparse.Namespace, log) -> int:
    from metetl.images.processing import ImageProcessor

    log.info("Команда process: input=%s, output=%s, num=%d, impl=%s, streaming=%s",
             args.input, args.output, args.num, args.impl, args.streaming)

    processor = ImageProcessor(output_root=args.output, impl=args.impl)
    indexed = ImageProcessor.load_indexed(args.input, args.num)
    if not indexed:
        log.error("Нечего обрабатывать")
        return 1
    suffix = "_streaming" if args.streaming else ""
    subdir = processor.make_run_subdir(suffix=suffix)

    start = time.perf_counter()
    if args.streaming:
        success = asyncio.run(processor.run_pipeline_streaming(indexed, subdir))
    else:
        success = asyncio.run(processor.run_pipeline(indexed, subdir))
    elapsed = time.perf_counter() - start

    log.info("Всего: %.2f с, обработано %d/%d -> %s",
             elapsed, success, len(indexed), subdir)
    return 0 if success > 0 else 1


def _cmd_analyze(args: argparse.Namespace, log) -> int:
    from metetl.analysis.aggregations import run_full_analysis

    log.info("Команда analyze: csv=%s, output-dir=%s", args.csv, args.output_dir)
    run_full_analysis(args.csv, args.output_dir)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    console_level = logging.DEBUG if args.verbose else logging.INFO
    log = configure(log_file=args.log_file, console_level=console_level)
    log.info("metetl %s starting: %s", __version__, args.command)

    try:
        if args.command == "prepare":
            return _cmd_prepare(args, log)
        if args.command == "process":
            return _cmd_process(args, log)
        if args.command == "analyze":
            return _cmd_analyze(args, log)
        parser.error(f"unknown command: {args.command}")
        return 2
    except KeyboardInterrupt:
        log.warning("Прервано пользователем")
        return 130
    except Exception:
        log.exception("Необработанная ошибка")
        return 1


if __name__ == "__main__":
    sys.exit(main())
