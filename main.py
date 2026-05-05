import argparse
import asyncio
import time

from image_processor import ImageProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab 4: асинхронная загрузка + параллельная свёртка")
    parser.add_argument('-n', '--count', type=int, default=5,
                        help="сколько картин скачать и обработать")
    parser.add_argument('--csv', default='MetObjects.csv',
                        help="путь к MetObjects.csv")
    parser.add_argument('--streaming', action='store_true',
                        help="использовать пайплайн на async-генераторах (бонус)")
    args = parser.parse_args()

    processor = ImageProcessor()

    start = time.perf_counter()
    if args.streaming:
        success, subdir = asyncio.run(
            processor.run_pipeline_streaming(args.csv, args.count))
    else:
        success, subdir = asyncio.run(
            processor.run_pipeline(args.csv, args.count))
    elapsed = time.perf_counter() - start

    print(f"\n[TIME] Всего: {elapsed:.2f} с, обработано {success}/{args.count}")
    print(f"[TIME] Результаты: {subdir}")


if __name__ == "__main__":
    main()
