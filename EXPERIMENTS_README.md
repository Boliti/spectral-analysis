# Воспроизведение экспериментов для ВКР

Тема: «Модуль для онлайн-платформы математического анализа спектральных данных».

## Добавленные скрипты

- `scripts/analyze_andor_slit_dataset.py` - анализ `.asc` спектров Andor по ширине входной щели и дифракционной решетке.
- `scripts/noise_robustness_experiment.py` - устойчивость классификации к гауссову шуму.
- `scripts/resolution_robustness_experiment.py` - устойчивость классификации к вычислительной имитации снижения спектрального разрешения.
- `scripts/hyperparameter_selection.py` - подбор гиперпараметров для классификации, регрессии и k-means.
- `scripts/benchmark_analysis_module.py` - оценка времени работы и пикового потребления памяти.
- `scripts/experiment_utils.py` - общие функции для чтения таблиц, моделей, метрик и сохранения простых PNG-графиков.

## Входные данные

Для экспериментов классификации нужен CSV/XLSX:

- одна строка = один спектр;
- столбец `target` содержит класс;
- спектральные точки находятся в числовых столбцах;
- служебные столбцы `sample_id`, `source_file`, `source_sheet` не используются как признаки.

Для регрессии нужен такой же CSV/XLSX, но `target` должен быть числовым.

Для Andor нужен каталог, ZIP-архив или отдельный `.asc` файл. Желательно, чтобы имя файла содержало ширину щели и решетку, например `sample_slit_50um_grating_600.asc`.

Если реального датасета нет, скрипты не создают вымышленные научные результаты: они завершаются с понятным сообщением о том, какой файл нужно передать.

## Команды PowerShell

```powershell
.\.venv\Scripts\python.exe scripts/analyze_andor_slit_dataset.py --input data/andor --output results/andor
.\.venv\Scripts\python.exe scripts/noise_robustness_experiment.py --input data/test1.csv --target target --output results/experiments
.\.venv\Scripts\python.exe scripts/resolution_robustness_experiment.py --input data/test1.csv --target target --output results/experiments
.\.venv\Scripts\python.exe scripts/hyperparameter_selection.py --input data/test1.csv --target target --task classification --output results/experiments
.\.venv\Scripts\python.exe scripts/benchmark_analysis_module.py --input data/test1.csv --target target --output results/experiments
```

Если используется системный Python с установленными зависимостями, можно заменить `.\.venv\Scripts\python.exe` на `python`.

## Результаты

Andor:

- `results/andor/andor_slit_metrics.csv`
- `results/andor/fwhm_vs_slit.png`
- `results/andor/snr_vs_slit.png`
- `results/andor/intensity_vs_slit.png`
- `results/andor/peak_position_vs_slit.png`
- `results/andor/example_spectra.png`

Эксперименты:

- `results/experiments/noise_robustness.csv`
- `results/experiments/noise_robustness_f1.png`
- `results/experiments/resolution_robustness.csv`
- `results/experiments/resolution_robustness_f1.png`
- `results/experiments/hyperparameter_selection.csv`
- `results/experiments/performance_benchmark.csv`

## Что вставлять в ВКР

- Сравнение классификаторов: таблицы `noise_robustness.csv`, `resolution_robustness.csv`, результаты `compare_classification` в интерфейсе `/analysis`.
- Сравнение регрессии: результаты `compare_regression` в интерфейсе `/analysis`, а также `hyperparameter_selection.csv` для регрессионной задачи.
- Кластеризация: метрики `silhouette_score`, `adjusted_rand_score`, `normalized_mutual_info_score`, `cluster_distribution` из интерфейса `/analysis` и таблиц сравнения.
- Влияние щели и решетки: `andor_slit_metrics.csv` и PNG-графики из `results/andor`.
- Устойчивость к шуму: `noise_robustness.csv` и `noise_robustness_f1.png`.
- Устойчивость к снижению разрешения: `resolution_robustness.csv` и `resolution_robustness_f1.png`.
- Быстродействие и память: `performance_benchmark.csv`.

## Примечания

Скрипт `resolution_robustness_experiment.py` моделирует ухудшение разрешения вычислительно: Gaussian smoothing, moving average и downsampling не являются реальными настройками спектрометра. Реальные экспериментальные параметры, например ширина щели и решетка, анализируются отдельно в `analyze_andor_slit_dataset.py`.
