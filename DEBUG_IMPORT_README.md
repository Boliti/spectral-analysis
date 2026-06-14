# Smoke-test импорта данных

Проверка использует существующий importer проекта из `services/analysis/dataset_importer.py`.

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test_importers.py --excel data/Raman_krov_SSZ-zdorovye.xlsx --zip data/Raman_slit_dependence_16_03_2026.zip
```

Для контрольных файлов ожидается:

- Excel: 100 спектров, 2000 признаков, классы `health` и `heart disease`.
- ZIP/ASC: 56 спектров, около 2000 признаков, metadata `slit_width_um` и `grating_lines_mm`.

Если Excel не читается, установите зависимости из `requirements.txt`; для `.xlsx` нужен `openpyxl`.
