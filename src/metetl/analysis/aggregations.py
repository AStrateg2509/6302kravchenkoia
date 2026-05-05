"""Анализ коллекции Met Museum (Lab 3) и построение графиков."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Generator, Hashable, Tuple

import matplotlib

matplotlib.use("Agg")  # без интерактивного бэкенда — для CLI/серверов
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.ndimage import uniform_filter1d

from metetl.logging_config import get_logger

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

_log = get_logger("analysis.aggregations")


_NORM_RULES: pd.DataFrame = pd.DataFrame([
    ("Culture",        "Объединить варианты: 'Greek' / 'Ancient Greek' / 'Greek, Attic'"),
    ("Culture",        "Иерархия: регион → культура → период"),
    ("Culture",        "Сгруппировать редкие (<0.1%) варианты в 'Other'"),
    ("Medium",         "Извлечь первичный материал (Oil on canvas → Oil)"),
    ("Medium",         "Стандартизировать регистр и пунктуацию"),
    ("Medium",         "Словарь синонимов (watercolor / watercolour)"),
    ("Classification", "Нормализовать ед./мн. число (Painting / Paintings)"),
    ("Classification", "Верхнеуровневые категории (Drawing, Sculpture, Textile…)"),
    ("Country",        "Привести к стандарту ISO 3166-1"),
    ("Country",        "Объединить исторические формы ('West Germany' → 'Germany')"),
    ("Department",     "Проверить соответствие официальной структуре отделов музея"),
    ("__generic__",    "Пропуски: импутация по связанным полям или категория 'Unknown'"),
    ("__generic__",    "Пороговая группировка: доля <0.5% → 'Other'"),
], columns=["Field", "Rule"])


# ---------------------------------------------------------------------
# Пайплайн чтения CSV (генераторы)
# ---------------------------------------------------------------------


class DataPipeline:
    """Три этапа обработки — три генератора. Файл читается в один проход."""

    def __init__(self, file_path: str, chunksize: int = 50_000) -> None:
        self.file_path = file_path
        self.chunksize = chunksize
        self.total_rows: int = 0
        self.valid_rows: int = 0

    def read_chunks(self) -> Generator[pd.DataFrame, None, None]:
        for chunk in pd.read_csv(
            self.file_path,
            chunksize=self.chunksize,
            dtype={
                "Object Begin Date": "float32",
                "Country": "category",
                "Department": "category",
                "Culture": "category",
                "Medium": "category",
                "Classification": "category",
            },
            usecols=["Object Begin Date", "Country", "Department",
                     "Culture", "Medium", "Classification"],
            na_values=["", "NULL", "null", "None"],
            low_memory=False,
        ):
            self.total_rows += len(chunk)
            yield chunk

    def filter_chunks(self, chunks: Generator[pd.DataFrame, None, None]
                       ) -> Generator[pd.DataFrame, None, None]:
        for chunk in chunks:
            mask = (
                chunk["Country"].notna()
                & chunk["Object Begin Date"].notna()
                & chunk["Object Begin Date"].between(-3000, 2024)
            )
            filtered = chunk.loc[mask]
            self.valid_rows += len(filtered)
            if not filtered.empty:
                yield filtered

    @staticmethod
    def aggregate_chunks(chunks: Generator[pd.DataFrame, None, None]
                          ) -> Generator[pd.DataFrame, None, None]:
        for chunk in chunks:
            yield chunk[["Country", "Object Begin Date"]]

    @staticmethod
    def merge_all(chunks: Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
        return pd.concat(chunks, ignore_index=True)

    def run(self) -> pd.DataFrame:
        _log.info("Запуск пайплайна обработки данных...")
        result = self.merge_all(
            self.aggregate_chunks(self.filter_chunks(self.read_chunks()))
        )
        _log.info("Прочитано строк: %d, после фильтрации: %d, уникальных стран: %d",
                  self.total_rows, self.valid_rows, result["Country"].nunique())
        return result


# ---------------------------------------------------------------------
# Анализатор: тяготение стран к эпохам
# ---------------------------------------------------------------------


class MetAnalyzer:
    def __init__(self, file_path: str, chunksize: int = 50_000) -> None:
        self.file_path = file_path
        self.chunksize = chunksize
        self._pipeline = DataPipeline(file_path, chunksize)
        self._data: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _calc_stats(s: pd.Series) -> pd.Series:
        n = len(s)
        if n < 2:
            return pd.Series({"count": n, "mean": np.nan, "ci_lo": np.nan,
                              "ci_hi": np.nan, "p025": np.nan, "p975": np.nan})
        mean = float(s.mean())
        std = float(s.std(ddof=1))
        ci_lo, ci_hi = scipy_stats.t.interval(0.95, df=n - 1, loc=mean,
                                               scale=std / np.sqrt(n))
        return pd.Series({
            "count": n,
            "mean": mean,
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "p025": float(s.quantile(0.025)),
            "p975": float(s.quantile(0.975)),
        })

    def analyze_top_countries(self, n_top: int = 10) -> pd.DataFrame:
        _log.info("Анализ топ-%d стран", n_top)
        self._data = self._pipeline.run()

        top_idx = self._data["Country"].value_counts().head(n_top).index
        data_top = (self._data[self._data["Country"].isin(top_idx)]
                    .assign(Country=lambda df: df["Country"].astype(str)))

        df = (data_top.groupby("Country")["Object Begin Date"]
              .apply(self._calc_stats).unstack().reset_index())

        for col in ("mean", "ci_lo", "ci_hi", "p025", "p975"):
            df[f"{col}_c"] = df[col] / 100.0

        return df.sort_values("count", ascending=False).reset_index(drop=True)

    @staticmethod
    def plot_country_centuries(df: pd.DataFrame, output_path: Path) -> Path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle("Анализ тяготения стран к эпохам (топ-10)",
                     fontsize=14, fontweight="bold")

        countries = df["Country"].values
        y_pos = np.arange(len(countries))
        means = df["mean_c"].values
        err_lo = means - df["ci_lo_c"].values
        err_hi = df["ci_hi_c"].values - means

        ax1.barh(y_pos, means, xerr=[err_lo, err_hi], capsize=5,
                 color="steelblue", alpha=0.75, error_kw={"elinewidth": 1.5})
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(countries)
        ax1.set_xlabel("Столетие")
        ax1.set_title("Среднее столетие создания\n(95% ДИ)")
        ax1.axvline(0, color="black", linewidth=0.6)
        ax1.grid(axis="x", alpha=0.3)

        p025 = df["p025_c"].values
        p975 = df["p975_c"].values
        for i, (lo, hi) in enumerate(zip(p025, p975)):
            ax2.hlines(i, lo, hi, color="darkorange", linewidth=4, alpha=0.7)
        ax2.plot(means, y_pos, "ro", markersize=7, label="Среднее", zorder=3)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(countries)
        ax2.set_xlabel("Столетие")
        ax2.set_title("95% интервал рассеяния")
        ax2.legend()
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        _log.info("Сохранён график: %s", output_path)
        return output_path

    def find_most_modern_country(self, min_objects: int = 10) -> Tuple[Hashable, float]:
        if self._data.empty:
            self._data = self._pipeline.run()
        agg = (self._data.groupby("Country", observed=True)["Object Begin Date"]
               .agg(mean_date="mean", n="count"))
        means_s = agg.loc[agg["n"] >= min_objects, "mean_date"]
        country = means_s.idxmax()
        return country, float(means_s.max())

    def plot_temporal_trend(self, country: str, output_path: Path,
                              window_size: int = 15, bin_width: int = 2) -> Path:
        dates = (self._data[self._data["Country"] == country]["Object Begin Date"]
                 .astype(float).sort_values().reset_index(drop=True))
        if dates.empty:
            _log.warning("Нет данных для страны %s", country)
            return output_path

        lo = int(np.floor(dates.min() / bin_width) * bin_width)
        hi = int(np.ceil(dates.max() / bin_width) * bin_width) + bin_width
        bins = np.arange(lo, hi + 1, bin_width)
        hist_vals, edges = np.histogram(dates.values, bins=bins)
        centers = pd.Series((edges[:-1] + edges[1:]) / 2.0)
        hist_s = pd.Series(hist_vals, dtype=float)
        smooth = pd.Series(uniform_filter1d(hist_s.values, size=window_size, mode="nearest"))

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(centers, hist_s, width=bin_width * 0.9, alpha=0.45,
               color="steelblue", label="Объектов за период")
        ax.plot(centers, smooth, "r-", linewidth=2,
                label=f"Скользящее среднее (окно={window_size})")
        ax.axvline(dates.mean(), color="green", linestyle="--", alpha=0.8,
                   label=f"Среднее: {dates.mean():.0f}")
        ax.axvline(dates.median(), color="orange", linestyle="--", alpha=0.8,
                   label=f"Медиана: {dates.median():.0f}")
        ax.set_xlabel("Год создания")
        ax.set_ylabel(f"Количество объектов (бин {bin_width} лет)")
        ax.set_title(f"Временной график — {country} (всего: {len(dates):,})")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        _log.info("Сохранён график: %s", output_path)
        return output_path


# ---------------------------------------------------------------------
# Анализатор качества категориальных полей
# ---------------------------------------------------------------------


class DataQualityAnalyzer:
    FIELDS = ("Department", "Culture", "Medium", "Classification", "Country")

    def __init__(self, file_path: str, chunksize: int = 50_000) -> None:
        self.file_path = file_path
        self.chunksize = chunksize

    def _read_chunks(self) -> Generator[pd.DataFrame, None, None]:
        for chunk in pd.read_csv(
            self.file_path, chunksize=self.chunksize, usecols=list(self.FIELDS),
            dtype={f: "object" for f in self.FIELDS},
            na_values=["", "NULL", "null", "None"],
        ):
            yield chunk

    def _count_chunks(self, chunks: Generator[pd.DataFrame, None, None]
                        ) -> Generator[pd.DataFrame, None, None]:
        for chunk in chunks:
            yield pd.concat([
                chunk[field].fillna("__NA__").value_counts()
                .rename("count").rename_axis("value")
                .reset_index().assign(field=field)
                for field in self.FIELDS
            ], ignore_index=True)

    @staticmethod
    def _gini(freq: pd.Series) -> float:
        freq = freq[freq > 0]
        if freq.empty:
            return 0.0
        n = len(freq)
        s = freq.sort_values().values
        idx = np.arange(1, n + 1)
        return float((2 * np.dot(idx, s)) / (n * s.sum()) - (n + 1) / n)

    @staticmethod
    def _entropy(freq: pd.Series) -> float:
        freq = freq[freq > 0]
        if freq.empty:
            return 0.0
        p = freq / freq.sum()
        h = float(-(p * np.log(p)).sum())
        h_max = np.log(len(p))
        return h / h_max if h_max > 0 else 0.0

    @staticmethod
    def _enc(freq: pd.Series) -> float:
        freq = freq[freq > 0]
        if freq.empty:
            return 0.0
        p = freq / freq.sum()
        enc = float(np.exp(-(p * np.log(p)).sum()))
        return enc / len(p)

    def _metrics_for_group(self, group: pd.DataFrame, total_rows: int) -> pd.Series:
        freq = group["count"]
        na_count = int(group.loc[group["value"] == "__NA__", "count"].sum())
        return pd.Series({
            "Categories": len(freq),
            "Missing %": round(na_count / total_rows * 100, 2),
            "Top cat %": round(float(freq.max()) / total_rows * 100, 2),
            "Gini": round(self._gini(freq), 4),
            "Entropy (norm)": round(self._entropy(freq), 4),
            "ENC (norm)": round(self._enc(freq), 4),
        })

    def analyze(self) -> pd.DataFrame:
        _log.info("Анализ качества категориальных полей")
        counted = pd.concat(self._count_chunks(self._read_chunks()), ignore_index=True)
        field_counts = counted.groupby(["field", "value"])["count"].sum().reset_index()
        total_rows = int(field_counts.loc[field_counts["field"] == self.FIELDS[0], "count"].sum())
        df = (field_counts.groupby("field")
              .apply(lambda g: self._metrics_for_group(g, total_rows))
              .reset_index().rename(columns={"field": "Field"})
              .sort_values("Gini", ascending=False).reset_index(drop=True))
        _log.info("Метрики качества:\n%s", df.to_string(index=False))
        return df

    @staticmethod
    def plot_heatmap(df: pd.DataFrame, output_path: Path) -> Path:
        metrics = ["Gini", "Entropy (norm)", "ENC (norm)"]
        hm_data = df.set_index("Field")[metrics]
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(hm_data, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0, vmax=1, linewidths=0.5,
                    cbar_kws={"label": "Значение (0–1)"}, ax=ax)
        ax.set_title("Тепловая карта качества категориальных полей")
        ax.set_ylabel("")
        plt.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        _log.info("Сохранён график: %s", output_path)
        return output_path

    @staticmethod
    def identify_problems(df: pd.DataFrame) -> pd.DataFrame:
        def _gen() -> Generator[pd.DataFrame, None, None]:
            yield pd.DataFrame(columns=["Field", "Issue"])
            mask = df["Gini"] > 0.7
            if mask.any():
                yield (df.loc[mask, ["Field", "Top cat %", "Gini"]]
                       .assign(Issue=("Топ-категория: "
                                       + df.loc[mask, "Top cat %"].map("{:.1f}%".format)
                                       + "; Gini="
                                       + df.loc[mask, "Gini"].map("{:.3f}".format)))
                       [["Field", "Issue"]])
            mask = df["Missing %"] > 10
            if mask.any():
                yield (df.loc[mask, ["Field", "Missing %"]]
                       .assign(Issue="Высокий % пропусков: "
                                      + df.loc[mask, "Missing %"].map("{:.1f}%".format))
                       [["Field", "Issue"]])
            mask = df["Entropy (norm)"] < 0.5
            if mask.any():
                yield (df.loc[mask, ["Field"]]
                       .assign(Issue="Низкая информативность распределения (энтропия < 0.5)"))
            mask = (df["Categories"] > 500) & (df["Gini"] > 0.6)
            if mask.any():
                yield (df.loc[mask, ["Field", "Categories"]]
                       .assign(Issue="Избыток редких категорий: "
                                      + df.loc[mask, "Categories"].astype(int).astype(str)
                                      + " уникальных значений")
                       [["Field", "Issue"]])

        return pd.concat(_gen(), ignore_index=True).sort_values("Field").reset_index(drop=True)


# ---------------------------------------------------------------------
# Точка входа CLI-команды analyze
# ---------------------------------------------------------------------


def run_full_analysis(csv_path: Path | str, output_dir: Path | str) -> None:
    """Полный прогон Lab 3: топ стран, временной тренд, качество данных. Графики сохраняются в output_dir."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    if not csv_path.exists():
        _log.error("CSV не найден: %s", csv_path)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = MetAnalyzer(str(csv_path))
    df_top = analyzer.analyze_top_countries(n_top=10)
    _log.info("Топ-10 стран:\n%s",
              df_top[["Country", "count", "mean_c", "ci_lo_c", "ci_hi_c"]]
              .to_string(index=False))
    analyzer.plot_country_centuries(df_top, output_dir / "top_countries.png")

    modern_country, avg_year = analyzer.find_most_modern_country()
    _log.info("Самая «современная» страна: %s (средний год: %.0f)", modern_country, avg_year)
    analyzer.plot_temporal_trend(str(modern_country),
                                   output_dir / "temporal_trend.png",
                                   window_size=5, bin_width=10)

    qa = DataQualityAnalyzer(str(csv_path))
    df_quality = qa.analyze()
    qa.plot_heatmap(df_quality, output_dir / "quality_heatmap.png")

    problems = qa.identify_problems(df_quality)
    if not problems.empty:
        _log.info("Идентифицированные проблемы:\n%s", problems.to_string(index=False))
    _log.info("Анализ завершён, графики -> %s", output_dir)
