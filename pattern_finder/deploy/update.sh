#!/usr/bin/env bash
# Обновление на сервере: свежий код из GitHub, библиотеки, перезапуск.
# Запуск из папки pattern_finder:  bash deploy/update.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
git pull
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q -x   # тесты не прошли — бот не перезапускается со сломанным кодом
sudo systemctl restart pattern-finder
echo "✅ Обновлено и перезапущено. Логи: journalctl -u pattern-finder -f"
