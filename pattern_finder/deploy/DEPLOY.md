# Pattern Finder на сервере 24/7

Пока бот запущен на твоём компьютере, он работает только когда компьютер включён.
На сервере бот работает круглосуточно, и автоскан не пропускает ни одной закрытой свечи.

## 1. Сервер

Нужен самый простой VPS: Ubuntu 22.04 или 24.04, 1 CPU, 1–2 ГБ памяти, 10+ ГБ диска.

⚠ **Важно: выбирай сервер в Европе или Азии, не в США.** Binance не отдаёт данные
серверам из США — бот не получит свечи.

После покупки у тебя будет IP-адрес сервера и пароль (или ключ) пользователя `root`.

## 2. Подключение

В Windows открой PowerShell:

```
ssh root@IP_СЕРВЕРА
```

Создай отдельного пользователя для бота (работать под root небезопасно):

```
adduser bot
usermod -aG sudo bot
su - bot
```

## 3. Код

```
git clone https://github.com/nomiseee69/candels.git
cd candels/pattern_finder
```

Если репозиторий приватный, git спросит логин и пароль: вместо пароля вставь
GitHub-токен (GitHub → Settings → Developer settings → Personal access tokens).

## 4. Установка

```
bash deploy/install.sh
nano .env
```

В `.env` впиши:

```
TELEGRAM_TOKEN=токен от BotFather
TELEGRAM_CHAT_ID=твой chat_id
```

Сохранить в nano: `Ctrl+O`, `Enter`, выйти: `Ctrl+X`.

## 5. Твои паттерны (по желанию)

Чтобы не загружать скрины заново, скопируй с компьютера `users.json` и папку `patterns`.
В PowerShell на компьютере, из папки `pattern_finder`:

```
scp users.json bot@IP_СЕРВЕРА:~/candels/pattern_finder/
scp -r patterns bot@IP_СЕРВЕРА:~/candels/pattern_finder/
```

## 6. Запуск

⚠ **Сначала останови бота на компьютере** (Ctrl+C). Два бота с одним токеном мешают
друг другу — Telegram отдаёт сообщения то одному, то другому.

```
sudo systemctl restart pattern-finder
journalctl -u pattern-finder -f
```

В логах должно быть «Бот запущен (обновление N)».
Выйти из логов: `Ctrl+C` (бот продолжит работать). Проверка: напиши боту в Telegram.

## 7. Обновления

Закоммить и запушь новую версию с компьютера (`git push`), потом на сервере:

```
cd ~/candels/pattern_finder
bash deploy/update.sh
```

Скрипт скачает код, прогонит тесты и перезапустит бота.

## Полезные команды

| Что | Команда |
| --- | --- |
| Статус бота | `sudo systemctl status pattern-finder` |
| Перезапуск | `sudo systemctl restart pattern-finder` |
| Остановить | `sudo systemctl stop pattern-finder` |
| Логи вживую | `journalctl -u pattern-finder -f` |
| Логи за час | `journalctl -u pattern-finder --since "1 hour ago"` |
