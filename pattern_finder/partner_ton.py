"""Read-only TON Mainnet verification through TonAPI. No signing or broadcast."""
import base64
import hashlib
import os
import re
import struct
from dataclasses import dataclass
from itertools import islice

import requests
from nacl.signing import VerifyKey
from pytoniq_core import Address, Cell, StateInit
from pytoniq_core.boc.deserialize import BocError
from pytoniq_core.boc.slice import SliceError

import partner_finance as finance

MASTER = "EQCxE6mUtQJKFnGfaROTKOt1lZbDiiX1kCixRv7Nw2Id_sDs"
WALLET_CODES = {
    "feb5ff6820e2ff0d9483e7e0d62c817d846789fb4ae580c878866d959dabd5c0": 64,
    "20834b7b72b112147e1b2fb457b84e74d1a30f04f737d4f62a668e9552d2b72f": 65,
}
PROVIDER = "https://tonapi.io"
PROOF_TTL = 600


def address(value):
    if not isinstance(value, str) or len(value) > 72:
        raise finance.Error("Некорректный TON-адрес.")
    try:
        parsed = Address(value)
        if parsed.wc != 0 or len(parsed.hash_part) != 32 or getattr(parsed, "is_test_only", False):
            raise ValueError
        return f"0:{parsed.hash_part.hex()}"
    except Exception:
        raise finance.Error("Нужен адрес TON Mainnet (workchain 0).") from None


def treasury():
    if os.getenv("PAYOUT_NETWORK", "TON_MAINNET") != "TON_MAINNET":
        raise finance.Error("Выплаты поддерживают только TON Mainnet.")
    owner = address(os.getenv("TON_TREASURY_ADDRESS", ""))
    if address(os.getenv("TON_PAYOUT_SOURCE_ADDRESS", "")) != owner:
        raise finance.Error("Кошелёк резерва и источник выплат должны совпадать.")
    return owner


def master():
    if address(os.getenv("USDT_TON_MASTER_ADDRESS", MASTER)) != address(MASTER):
        raise finance.Error("Разрешён только официальный USDT на TON.")
    return address(MASTER)


def hash_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise finance.Error("TXID должен состоять из 64 шестнадцатеричных символов.")
    return value.lower()


def get(path):
    # Pin the trusted Mainnet provider and never forward its token via redirects.
    if os.getenv("TON_API_PROVIDER_URL", PROVIDER).rstrip("/") != PROVIDER:
        raise finance.Error("Неподдерживаемый провайдер TON Mainnet.")
    api_key = os.getenv("TON_API_PROVIDER_API_KEY")
    try:
        response = requests.get(PROVIDER + "/v2/" + path, timeout=(3, 8), allow_redirects=False,
                                headers={"Authorization": "Bearer " + api_key} if api_key else {})
        if response.status_code == 404:
            raise finance.Error("TON ещё не подтвердил данные. Проверь адрес или повтори позднее.")
        if response.status_code != 200:
            raise ValueError
        return response.json()
    except (requests.RequestException, ValueError) as error:
        if isinstance(error, finance.Error):
            raise
        raise finance.Error("Данные TON временно недоступны.") from None


def wallet_key(owner):
    return bytes.fromhex(get(f"accounts/{owner}/publickey")["public_key"].removeprefix("0x"))


def proof_digest(owner, domain, at, payload):
    name = domain.encode("utf-8")
    content = (b"ton-proof-item-v2/" + struct.pack(">i", 0) + bytes.fromhex(owner[2:])
               + struct.pack("<I", len(name)) + name + struct.pack("<Q", at) + payload.encode())
    return hashlib.sha256(b"\xff\xffton-connect" + hashlib.sha256(content).digest()).digest()


def proof_owner(account, proof, domain, issued, now):
    try:
        if account.get("chain") != "-239" or not domain:
            raise ValueError
        owner = address(account["address"])
        stamp = proof["timestamp"]
        if type(stamp) is not int or not issued - 30 <= stamp <= now + 30 or now - stamp > PROOF_TTL:
            raise ValueError
        if proof["domain"] != {"value": domain, "lengthBytes": len(domain.encode("utf-8"))}:
            raise ValueError
        state = Cell.one_from_boc(base64.b64decode(account["walletStateInit"], validate=True))
        if state.hash.hex() != owner[2:]:
            raise ValueError
        parsed = StateInit.deserialize(state.begin_parse())
        offset = WALLET_CODES.get(parsed.code.hash.hex())
        if offset is None:
            raise ValueError
        data = parsed.data.begin_parse()
        data.skip_bits(offset)
        key = data.load_bytes(32)
        declared = account.get("publicKey")
        if declared and bytes.fromhex(declared) != key:
            raise ValueError
        signature = base64.b64decode(proof["signature"], validate=True)
        VerifyKey(key).verify(proof_digest(owner, domain, stamp, proof["payload"]), signature)
        # The proof does not sign a chain ID. Require the reviewed key on an active Mainnet wallet.
        if wallet_key(owner) != key:
            raise ValueError
        return owner
    except Exception:
        raise finance.Error("Кошелёк TON Mainnet не подтверждён. Используй активный Wallet V4R2 или V5R1.") from None


def jetton_wallet(owner):
    try:
        row = get(f"accounts/{owner}/jettons/{master()}")
        if address(row["jetton"]["address"]) != master() or str(row["jetton"]["decimals"]) != "6":
            raise finance.Error("Токен или точность USDT не подтверждены.")
        balance = quantity(row["balance"], allow_zero=True)
        return address(row["wallet_address"]["address"]), balance
    except (KeyError, TypeError, AttributeError):
        raise finance.Error("Индексатор не подтвердил баланс USDT.") from None


def quantity(value, *, allow_zero=False):
    if type(value) not in (int, str) or not re.fullmatch(r"[0-9]{1,17}", str(value)):
        raise finance.Error("TON вернул недопустимую сумму.")
    units = int(value)
    if not units and not allow_zero:
        raise finance.Error("Получена нулевая сумма USDT.")
    return units


@dataclass(frozen=True)
class Transfer:
    root: str
    receipt: str
    source: str
    destination: str
    amount: int
    at: int


def _internal_outgoing(tx):
    if not tx.get("raw"):
        return [hash_id(msg["hash"]) for msg in tx["out_msgs"] if msg.get("msg_type") == "int_msg"]
    # TonAPI traces can omit out_msgs; each raw transaction retains the original message cells.
    cell = Cell.one_from_boc(bytes.fromhex(tx["raw"]))
    transaction = cell.begin_parse()
    if cell.hash.hex() != hash_id(tx["hash"]) or transaction.load_uint(4) != 0b0111:
        raise finance.Error("Данные транзакции TON не соответствуют её хешу.")
    transaction.skip_bits(256 + 64 + 256 + 64 + 32)  # account, lt, previous hash and lt, timestamp
    count = transaction.load_uint(15)
    messages = transaction.load_ref().begin_parse()
    messages.load_maybe_ref()  # inbound message
    outgoing = messages.load_dict(15, value_deserializer=lambda entry: entry.load_ref()) or {}
    if len(outgoing) != count:
        raise finance.Error("Неполный список исходящих сообщений TON.")
    return [msg.hash.hex() for msg in outgoing.values() if not msg.begin_parse().preload_bit()]


def _transactions(trace):
    tx = trace["transaction"]
    children = trace.get("children", [])
    if trace.get("emulated") is True:
        raise finance.Error("Транзакция ещё не подтверждена TON.")
    outgoing = _internal_outgoing(tx)
    inbound = [hash_id(c["transaction"]["in_msg"]["hash"]) for c in children]
    if sorted(outgoing) != sorted(inbound):
        raise finance.Error("Индексатор ещё не получил всю цепочку перевода.")
    yield tx, outgoing
    for child in children:
        yield from _transactions(child)


def _successful(tx):
    return (tx.get("success") is True and tx.get("aborted") is False
            and tx.get("destroyed") is False and tx["in_msg"].get("bounced") is False
            and not tx.get("bounce_phase"))


def verify_transfer(txid, source, destination, expected=None, since=0):
    """Verify the canonical trace and USDT recipient wallet's internal transfer."""
    txid = hash_id(txid)
    source, destination = address(source), address(destination)
    try:
        sender_wallet, _ = jetton_wallet(source)
        recipient_wallet, _ = jetton_wallet(destination)
        trace = get("traces/" + txid)
        root = trace["transaction"]
        canonical = hash_id(root["hash"])
        if address(root["account"]["address"]) != source or not _successful(root):
            raise finance.Error("Исходная транзакция кошелька отправителя не подтверждена.")
        txs = list(islice(_transactions(trace), 33))
        if len(txs) > 32:
            raise finance.Error("Для сверки нужен отдельный перевод USDT без пакетных операций.")
        if txid not in {hash_id(tx["hash"]) for tx, _ in txs}:
            raise finance.Error("TXID не входит в полученную цепочку.")
        matches = []
        for tx, _ in txs:
            if address(tx["account"]["address"]) != recipient_wallet:
                continue
            msg = tx["in_msg"]
            if not msg.get("source") or address(msg["source"]["address"]) != sender_wallet:
                continue
            body = Cell.one_from_boc(bytes.fromhex(msg["raw_body"])).begin_parse()
            if body.load_uint(32) != 0x178D4519:  # TEP-74 internal_transfer
                continue
            body.skip_bits(64)  # query_id
            paid = quantity(body.load_coins())
            initiator = body.load_address()
            if initiator is None or address(initiator.to_str(is_user_friendly=False)) != source:
                continue
            if not _successful(tx) or not any(
                    _successful(parent) and address(parent["account"]["address"]) == sender_wallet
                    and hash_id(msg["hash"]) in outgoing for parent, outgoing in txs):
                raise finance.Error("Перевод USDT неуспешен или был возвращён.")
            matches.append(Transfer(canonical, hash_id(tx["hash"]), source, destination, paid, int(tx["utime"])))
        if len(matches) != 1:
            raise finance.Error("Нужен один прямой перевод официального USDT на заданный адрес.")
        transfer = matches[0]
        if (expected is not None and transfer.amount != expected) or transfer.at < since:
            raise finance.Error("Сумма или дата перевода не соответствует операции.")
        return transfer
    except (KeyError, TypeError, ValueError, AttributeError, BocError, SliceError, IndexError) as error:
        if isinstance(error, finance.Error):
            raise
        raise finance.Error("Индексатор не подтвердил запрошенный перевод USDT.") from None
