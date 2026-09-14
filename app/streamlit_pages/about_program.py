import streamlit as st
import base64
import re
import markdown
import mimetypes
from pathlib import Path

from app.version import APP_VERSION

# Необходимые пути
# parents[0] -> streamlit
# parents[1] -> app
# parents[2] -> Infill_drilling
PROJECT_ROOT = Path(__file__).resolve().parents[2]
README_PATH = PROJECT_ROOT / "README.md"
DIAGRAMS_DIR = PROJECT_ROOT / "diagrams"
LICENSE_PATH = PROJECT_ROOT / "LICENSE"


# =========================================================
# Работа с файлами
# =========================================================


def file_to_base64(file_path: Path) -> str:
    """Преобразование файла в Base64."""
    return base64.b64encode(
        file_path.read_bytes()
    ).decode("utf-8")


def get_mime_type(file_path: Path) -> str:
    """Определение MIME-типа файла."""

    mime_type, _ = mimetypes.guess_type(
        str(file_path)
    )

    return mime_type or "application/octet-stream"


# =========================================================
# Обработка изображений
# =========================================================

def process_images(markdown_text: str) -> str:
    """
    Заменяет локальные Markdown-картинки
    на HTML с встроенным изображением.
    """

    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    def replace_markdown_image(match):

        alt_text = match.group(1)
        relative_path = match.group(2).strip().strip("<>")

        image_path = PROJECT_ROOT / relative_path

        if not image_path.exists():
            return match.group(0)

        try:
            mime_type = get_mime_type(image_path)
            image_data = file_to_base64(image_path)

            return (
                f'<img '
                f'src="data:{mime_type};base64,{image_data}" '
                f'alt="{alt_text}" '
                f'style="'
                f'max-width:100%; '
                f'height:auto; '
                f'display:block; '
                f'margin:20px auto;'
                f'">'
            )

        except Exception as e:
            st.warning(
                f"Не удалось загрузить изображение "
                f"{relative_path}: {e}"
            )

            return match.group(0)

    return re.sub(
        image_pattern,
        replace_markdown_image,
        markdown_text
    )


def process_html_images(markdown_text: str) -> str:
    """
    Обрабатывает изображения, заданные непосредственно
    через HTML:

        <img src="diagrams/image.png">
    """

    html_pattern = r'<img([^>]+)src=["\']([^"\']+)["\']([^>]*)>'

    def replace_html_image(match):

        before_src = match.group(1)
        relative_path = match.group(2)
        after_src = match.group(3)

        # Внешние картинки не трогаем
        if (
                relative_path.startswith("http://")
                or relative_path.startswith("https://")
                or relative_path.startswith("data:")
        ):
            return match.group(0)

        image_path = PROJECT_ROOT / relative_path

        if not image_path.exists():
            return match.group(0)

        try:
            mime_type = get_mime_type(image_path)
            image_data = file_to_base64(image_path)

            return (
                f'<img'
                f'{before_src}'
                f'src="data:{mime_type};base64,{image_data}"'
                f'{after_src}>'
            )

        except Exception:
            return match.group(0)

    return re.sub(
        html_pattern,
        replace_html_image,
        markdown_text,
        flags=re.IGNORECASE
    )


# =========================================================
# Обработка локальных ссылок
# =========================================================

def process_links(markdown_text: str) -> str:
    """
    Обрабатывает локальные ссылки README.
    Внешние ссылки не изменяются.
    """

    link_pattern = r'(?<!!)\[([^\]]+)\]\(([^)]+)\)'

    def replace_link(match):

        link_text = match.group(1)
        target = match.group(2).strip()

        # Якоря и внешние ссылки оставляем как есть
        if target.startswith(
                ("#", "http://", "https://", "mailto:")
        ):
            return match.group(0)

        # -------------------------------------------------
        # GitHub-style ссылки
        # -------------------------------------------------
        target_without_anchor = target.split("#")[0]

        local_path = (
                PROJECT_ROOT /
                target_without_anchor
        )

        if not local_path.exists():
            return match.group(0)

        # -------------------------------------------------
        # LICENSE
        # -------------------------------------------------

        try:
            if local_path.resolve() == LICENSE_PATH.resolve():
                return (f'<a href="#license">' f'{link_text}' f'</a>')
        except OSError:
            pass

        # -------------------------------------------------
        # Остальные локальные файлы оставляем как есть
        return match.group(0)

    return re.sub(
        link_pattern,
        replace_link,
        markdown_text
    )


# =========================================================
# Удаление содержания
# =========================================================
def remove_toc(markdown_text: str) -> str:
    """ Удаляет раздел 'Содержание' и 'Лицензия' из README.
    Сам README.md при этом не изменяется. """
    # Удаляем содержание
    markdown_text = re.sub(
        r'## 📚 Содержание.*?(?=\n## |\Z)',
        '',
        markdown_text,
        flags=re.DOTALL,
    )

    # Удаляем лицензию до конца README
    markdown_text = re.sub(
        r'## 📄 Лицензия.*\Z',
        '',
        markdown_text,
        flags=re.DOTALL,
    )

    return markdown_text

# =========================================================
# README
# =========================================================

def process_readme(markdown_text: str) -> str:
    markdown_text = remove_toc(markdown_text)
    markdown_text = process_images(markdown_text)
    markdown_text = process_html_images(markdown_text)
    markdown_text = process_links(markdown_text)

    markdown_text = markdown_text.replace(
        "(#-",
        "(#"
    )

    return markdown_text


# =========================================================
# Страница "О программе"
# =========================================================

def show():
    """Страница «О программе»."""

    # -----------------------------------------------------
    # Проверка README
    # -----------------------------------------------------

    if not README_PATH.exists():
        st.error(
            f"README.md не найден:\n\n"
            f"{README_PATH}"
        )

        return

    # -----------------------------------------------------
    # Читаем README
    # -----------------------------------------------------

    try:

        readme = README_PATH.read_text(
            encoding="utf-8"
        )

    except Exception as e:

        st.error(
            f"Не удалось прочитать README.md:\n\n{e}"
        )

        return

    # -----------------------------------------------------
    # Обрабатываем README
    # -----------------------------------------------------
    readme = process_readme(
        readme
    )

    # -----------------------------------------------------
    # Показываем README
    # -----------------------------------------------------
    st.markdown(
        readme,
        unsafe_allow_html=True,
    )
