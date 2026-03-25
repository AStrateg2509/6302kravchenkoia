def timeit(func):
    """
    Декоратор для замера времени выполнения функции.
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIME] {func.__name__} выполнена за {end - start:.4f} секунд")
        return result

    return wrapper
