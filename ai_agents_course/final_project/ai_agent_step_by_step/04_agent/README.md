# MLE Agents v2 - Langflow

Миграция MLE Agents с Google ADK на Langflow для визуальной оркестрации ML-агентов.

## Структура проекта

```
mle_agents_v2/
├── langflow_components/     # Custom Langflow компоненты
│   ├── code_executor/       # Code Executor компоненты
│   │   ├── executor_component.py    # ML Code Executor с feedback loop
│   │   └── pipeline_component.py    # Pipeline Orchestrator
│   ├── rag/                 # RAG компоненты
│   │   ├── retriever_component.py   # Hybrid RAG Retriever
│   │   ├── retriever_backend.py     # Backend логика RAG
│   │   └── utils.py                 # RAG утилиты
│   └── utils/               # Общие утилиты
├── data/                    # Данные для обучения
├── tickets/                 # Тикеты задач
├── workspace/               # Рабочая директория для выполнения кода
├── requirements.txt         # Python зависимости
├── run.py                   # Скрипт запуска Langflow
├── Dockerfile               # Docker образ
└── docker-compose.yml       # Docker Compose конфигурация
```

## Компоненты

### 1. Hybrid RAG Retriever
- **display_name**: Hybrid RAG Retriever
- **Поиск**: BM25 + Semantic (FAISS) + RRF reranking
- **Inputs**:
  - `query` - поисковый запрос
  - `k` - количество результатов
  - `search_type` - тип поиска (hybrid/semantic/bm25)
  - `storage_path` - путь к RAG хранилищу
- **Outputs**:
  - `results` - список Data объектов с результатами
  - `context` - форматированный контекст для промпта

### 2. ML Code Executor
- **display_name**: ML Code Executor
- **Функционал**:
  - Генерация Python кода через LLM
  - Выполнение кода в subprocess
  - Feedback loop для автоисправления ошибок
  - Детекция артефактов (log файлы, модели)
- **Inputs**:
  - `task` - описание задачи
  - `context` - контекст от предыдущего шага
  - `max_attempts` - макс. попыток (default: 5)
  - `timeout` - таймаут выполнения (default: 1800s)
  - `model` - LLM модель для генерации
- **Outputs**:
  - `result` - результат выполнения
  - `log_path` - путь к log файлу

### 3. ML Pipeline Orchestrator
- **display_name**: ML Pipeline Orchestrator
- **Функционал**:
  - Координация multi-step пайплайнов
  - Передача контекста между шагами
  - Шаблоны задач (EDA → Train → Predict)
- **Inputs**:
  - `data_path` - путь к данным
  - `target_column` - целевая колонка
  - `pipeline_config` - JSON конфигурация пайплайна

## Установка

### Локально

```bash
# Создать venv (Python 3.10-3.13)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Установить зависимости
pip install -r requirements.txt

# Скопировать .env
cp .env.example .env
# Отредактировать .env с вашим API ключом

# Запустить Langflow
python run.py
```

### Docker

```bash
# Собрать и запустить
docker-compose up --build

# UI доступен на http://localhost:7860
```

## Использование

1. Откройте http://localhost:7860
2. Создайте новый Flow
3. Добавьте компоненты:
   - **Hybrid RAG Retriever** → поиск примеров кода
   - **ML Code Executor** → генерация и выполнение кода
   - **LLM** → для генерации кода (Anthropic/OpenAI)
4. Соедините компоненты
5. Запустите Flow

## Пример Flow: EDA Analysis

```
[Chat Input] → [Hybrid RAG Retriever] → [ML Code Executor] → [Chat Output]
                                              ↑
                                        [LLM Model]
```

Задача: "Проведи EDA анализ данных из data/train.csv"

Результат:
- Код сгенерирован и выполнен
- Логи сохранены в workspace/eda_analysis.log
- Визуализации сохранены как артефакты

## Миграция с Google ADK

| Old (google-adk) | New (Langflow) |
|------------------|----------------|
| `Agent` + `tools` | `Component` с `inputs`/`outputs` |
| `FunctionTool` | Method в Component |
| `InMemoryRunner` | Langflow runtime |
| `AgentPipeline` | Visual flow в UI |
| `SessionManager` | Встроенный session manager |

## Требования

- Python 3.10-3.13
- Ollama (для эмбеддингов BGE-M3)
- OpenRouter API ключ (или другой LLM провайдер)

## Лицензия

MIT
