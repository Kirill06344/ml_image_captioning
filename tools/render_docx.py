"""Обновить оглавление DOCX и создать PDF через локальный LibreOffice и python3-uno.

Запускать Python, где установлен модуль uno (обычно системный python3):
    python3 tools/render_docx.py --input 'docs/report/Курсовая работа — Тесленко.docx'

DOCX обновляется на месте, PDF сохраняется рядом. Ничего не публикуется.
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from pathlib import Path


def normalize_core_relationship(source):
    """Repair LibreOffice's nonstandard core-properties relationship without altering layout/fields."""
    source = Path(source)
    legacy = b'http://schemas.openxmlformats.org/officedocument/2006/relationships/metadata/core-properties'
    standard = b'http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties'
    with zipfile.ZipFile(source) as archive:
        relationships = archive.read('_rels/.rels')
        if legacy not in relationships:
            return
        entries = [(item, archive.read(item.filename)) for item in archive.infolist()]
    with tempfile.NamedTemporaryFile(dir=source.parent, suffix='.docx', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, 'w') as archive:
            for item, data in entries:
                if item.filename == '_rels/.rels':
                    data = data.replace(legacy, standard)
                archive.writestr(item, data)
        os.replace(temporary, source)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--pdf', type=Path)
    args = p.parse_args()
    source = args.input.resolve(strict=True)
    pdf = args.pdf.resolve() if args.pdf else source.with_suffix('.pdf')
    binary = shutil.which('libreoffice') or shutil.which('soffice')
    if not binary:
        p.error('Нужен LibreOffice Writer + Math. Проверьте систему или изолированный контейнер.')
    try:
        import uno
        from com.sun.star.beans import PropertyValue
    except ImportError:
        p.error('Нужен python3-uno; запустите системным Python с этим модулем, а не произвольной .venv.')

    def prop(name, value):
        item = PropertyValue()
        item.Name, item.Value = name, value
        return item

    with tempfile.TemporaryDirectory(prefix='captioning-lo-') as temporary:
        temporary = Path(temporary)
        pipe = 'captioning_' + uuid.uuid4().hex
        address = f'pipe,name={pipe};urp;StarOffice.ComponentContext'
        process = subprocess.Popen([binary, '-env:UserInstallation=' + (temporary / 'profile').as_uri(),
                                    '--headless', '--norestore', '--nodefault', '--accept=' + address])
        desktop = doc = None
        try:
            local = uno.getComponentContext()
            resolver = local.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', local)
            for _ in range(100):
                try:
                    context = resolver.resolve('uno:' + address)
                    break
                except Exception:
                    if process.poll() is not None:
                        raise RuntimeError('LibreOffice завершился до подключения')
                    time.sleep(.1)
            else:
                raise RuntimeError('Не удалось подключиться к LibreOffice за 10 секунд')
            desktop = context.ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop', context)
            doc = desktop.loadComponentFromURL(source.as_uri(), '_blank', 0,
                                               (prop('Hidden', True), prop('UpdateDocMode', 3), prop('MacroExecutionMode', 4)))
            if doc is None:
                raise RuntimeError('LibreOffice не открыл DOCX; проверьте наличие Writer')
            doc.refresh()
            doc.getTextFields().refresh()
            indexes = doc.getDocumentIndexes()
            for i in range(indexes.getCount()):
                indexes.getByIndex(i).update()
            doc.getTextFields().refresh()
            saved = temporary / 'updated.docx'
            doc.storeToURL(saved.as_uri(), (prop('FilterName', 'Office Open XML Text'), prop('Overwrite', True)))
            pdf.parent.mkdir(parents=True, exist_ok=True)
            doc.storeToURL(pdf.as_uri(), (prop('FilterName', 'writer_pdf_Export'), prop('Overwrite', True)))
            doc.close(True)
            doc = None
            # Сначала копия рядом с назначением: os.replace остаётся атомарным на одном FS.
            with tempfile.NamedTemporaryFile(dir=source.parent, suffix='.docx', delete=False) as stream:
                replacement = Path(stream.name)
                stream.write(saved.read_bytes())
            os.replace(replacement, source)
            normalize_core_relationship(source)
            print(source)
            print(pdf)
        finally:
            if doc is not None:
                doc.close(True)
            if desktop is not None:
                desktop.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)


if __name__ == '__main__':
    main()
