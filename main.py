from image_processor import ImageProcessor


def main():
    """
    Основная функция для демонстрации работы классов.
    """
    processor = ImageProcessor()

    artwork = processor.download_image('MetObjects.csv')

    if artwork:
        print("\n" + str(artwork) + "\n")

        # Обрабатываем изображение
        results = processor.process_image(artwork)

        # Сохраняем результаты
        if results:
            object_id = artwork.metadata.get('objectID', 'unknown')
            processor.save_results(results, f"painting_{object_id}")

        # Демонстрируем полиморфизм
        processor.demonstrate_polymorphism()
    else:
        print("Не удалось загрузить изображение")


if __name__ == "__main__":
    main()
