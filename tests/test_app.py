from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_without_exceptions():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    assert not app.exception
    assert [title.value for title in app.title] == ["PDF RAG Assistant"]
    assert len(app.get("file_uploader")) == 1
    assert len(app.get("chat_input")) == 1


def test_app_initial_upload_and_chat_controls():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)

    parser = app.selectbox[0]
    assert parser.options == ["pypdf", "pdfplumber"]
    assert parser.value == "pypdf"

    chat_input = app.get("chat_input")[0]
    assert chat_input.disabled
