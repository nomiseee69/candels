#!/usr/bin/env bash
# Установка Pattern Finder на сервер с Ubuntu.
# Запуск из папки pattern_finder:  bash deploy/install.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(whoami)"
echo "▶ Папка проекта: $DIR, пользователь: $USER_NAME"

echo "▶ Ставлю Python"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git build-essential

echo "▶ Виртуальное окружение и библиотеки"
cd "$DIR"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if [ ! -f .env ]; then
  cp deploy/env.example .env
  echo "⚠ Создал .env — заполни токен:  nano .env"
fi

echo "▶ Служба systemd (бот работает 24/7 и сам поднимается после перезагрузки)"
sed -e "s|__USER__|$USER_NAME|g" -e "s|__DIR__|$DIR|g" deploy/pattern-finder.service \
  | sudo tee /etc/systemd/system/pattern-finder.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable pattern-finder

echo
echo "✅ Готово. Дальше:"
echo "   nano .env                              — заполни токен и chat_id (ADMIN_IDS — твой id)"
echo "   .venv/bin/python setup_brain.py        — нейросеть для понимания слов (по желанию, ~2 ГБ)"
echo "   sudo systemctl restart pattern-finder  — запустить / перезапустить"
echo "   journalctl -u pattern-finder -f        — смотреть логи (выход: Ctrl+C)"
