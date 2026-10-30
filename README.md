## Vid2 setup

Run these commands from **Git Bash**. The `--python` option makes every
install target Vid2's virtual environment, even when the shell has not
activated it yet.

```bash
cd "/g/gen ai sheriyans/Vid2"
uv venv --python 3.14 .venv
source .venv/Scripts/activate
uv pip install --upgrade pip
uv pip install -e .
```

Verify the environment and all Vid2 imports:

```bash
python -c "import sys; print(sys.executable)"
python -c "from dotenv import load_dotenv; from langchain_google_genai import ChatGoogleGenerativeAI; from langchain_community.vectorstores import Chroma; from langchain_huggingface import HuggingFaceEmbeddings; print('Vid2 dependencies OK')"
```

Run the examples from the Vid2 directory so relative document and vector-store
paths resolve correctly:

```bash
python main.py
python vector-store/db.py
```

If activation is unavailable in a new shell, use the environment explicitly:

```bash
uv pip install --python .venv/Scripts/python.exe -e .
.venv/Scripts/python.exe main.py
```