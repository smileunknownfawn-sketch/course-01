# Як запустити dashboard

## 1. Локально на Windows

Відкрий PowerShell або Terminal у папці проєкту.

### Перший запуск

Створи віртуальне середовище:

~~~powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
~~~

Завантаж актуальні відкриті історичні дані:

~~~powershell
python scripts/download_kaggle.py
~~~

Побудуй нормалізовані таблиці та перевір якість:

~~~powershell
python scripts/build_processed_kaggle.py
python scripts/data_quality_report.py
python scripts/self_improve.py
python scripts/build_dashboard_data.py
~~~

Запусти сайт:

~~~powershell
streamlit run streamlit_app.py
~~~

Streamlit покаже локальну адресу, зазвичай:

~~~text
http://localhost:8501
~~~

Відкрий її у браузері.

## 2. Наступні локальні оновлення

Для повного оновлення даних:

~~~powershell
python scripts/download_kaggle.py
python scripts/build_processed_kaggle.py
python scripts/data_quality_report.py
python scripts/self_improve.py
python scripts/build_dashboard_data.py
~~~

Якщо Streamlit уже працює, сторінку достатньо оновити.

## 3. Онлайн-режим

Репозиторій підготовлений для Streamlit Community Cloud.

Entry point (головний файл):

~~~text
streamlit_app.py
~~~

Після підключення GitHub-репозиторію до Streamlit Cloud застосунок читає
агреговані файли з:

~~~text
data/dashboard/
~~~

GitHub Actions виконує контрольований learning cycle щотижня,
будує новий dashboard snapshot і публікує його в репозиторій.
Після нового коміту онлайн-застосунок отримує оновлені дані.

## 4. Що оновлюється автоматично

- свіжий історичний Kaggle snapshot;
- очищення та нормалізація;
- quality gate (контроль якості);
- навчальна вибірка;
- challenger-модель;
- порівняння з baseline/champion;
- dashboard snapshot;
- дата останнього оновлення;
- статистика по областях;
- ML-метрики та стан готовності.

Карта 25 адміністративних регіонів зберігається в `data/geo/ukraine_admin1.geojson`,
тому вона працює без доступу до зовнішнього картографічного сервера. Межі:
[ukraine-geo-data](https://github.com/darmat1/ukraine-geo-data), дані OpenStreetMap
(ліцензія ODbL). Натискання області на карті або рядка таблиці змінює вибір для
всіх вкладок. Київ як окреме місто доступний у списку областей, але набір
контурів не містить його окремої геометрії.

Показник «Узгоджена історична частка» — середнє відсоткових часток Kaggle і
VIINA за вибраний період. Кількість записів цих джерел показано окремо, адже
вони мають різні методики та різні часові межі. Тривоги не враховуються як удари.

Вкладка «Збиття» використовує окрему таблицю
`data/dashboard/interception_by_type_daily.csv`: за типом цілі та датою вона
показує кількість запущених, заявлених збитих і різницю між ними. До частки
збиття входять лише записи, для яких обидві кількості коректні. Різниця має
назву «без підтвердженого збиття» — це **не** кількість влучань. Дані збиття
доступні лише в загальноукраїнському розрізі, бо прив'язка запису атаки до
області не вказує місця збиття окремої цілі. Фільтр дати діє на цю вкладку;
фільтр області не змінює загальноукраїнських чисел.

## 5. Що НЕ є live-моніторингом

Dashboard не є системою оперативного спостереження.

Він не показує:
- поточні маршрути БпЛА/ракет;
- точні майбутні цілі;
- координати запусків;
- оперативне переміщення засобів;
- точний час майбутньої атаки.

Оновлюються історичні агреговані дані та результати їх аналізу.

## 6. Якщо сайт показує, що snapshot відсутній

Запусти:

~~~powershell
python scripts/build_dashboard_data.py
~~~

або відкрий GitHub Actions і вручну запусти workflow
`weekly-learning-cycle`.
