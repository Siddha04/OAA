from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .assistant import PersonalAssistant
from .config import GenerationConfig
from .engine import Engine
from .memory import MemoryStore
from .rag import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genai",
        description="OAA local personal AI CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_generation_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--max-tokens", type=int, default=32)
        command.add_argument("--temperature", type=float, default=0.0)
        command.add_argument("--top-k", type=int, default=0)
        command.add_argument("--top-p", type=float, default=1.0)
        command.add_argument("--repetition-penalty", type=float, default=1.0)
        command.add_argument("--seed", type=int, default=0)

    def add_chunk_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--chunk-size", type=int, default=1000)
        command.add_argument("--chunk-overlap", type=int, default=150)

    ask = subparsers.add_parser("ask", help="run one local inference request")
    ask.add_argument("--model", required=True)
    ask.add_argument(
        "--system",
        default="You are OAA, a local personal AI assistant.",
    )
    ask.add_argument("prompt")
    add_generation_options(ask)

    chat = subparsers.add_parser("chat", help="start an interactive local chat")
    chat.add_argument("--model", required=True)
    chat.add_argument(
        "--system",
        default="You are OAA, a local personal AI assistant.",
    )
    chat.add_argument("--max-history-messages", type=int, default=32)
    chat.add_argument("--docs", help="optional local directory to use for RAG")
    chat.add_argument("--rag-top-k", type=int, default=3)
    chat.add_argument("--rag-max-context-chars", type=int, default=512)
    chat.add_argument(
        "--memory-db",
        help="optional SQLite file for explicit saved-memory recall",
    )
    add_chunk_options(chat)
    add_generation_options(chat)

    search = subparsers.add_parser(
        "search", help="search local documents without loading a language model"
    )
    search.add_argument("--docs", required=True, help="local document directory")
    search.add_argument("--top-k", type=int, default=5, help="maximum chunks to show")
    search.add_argument("--min-score", type=float, default=0.05)
    add_chunk_options(search)
    search.add_argument("query")

    memory = subparsers.add_parser(
        "memory", help="manage explicitly saved local memories"
    )
    memory_actions = memory.add_subparsers(dest="memory_action", required=True)

    def add_memory_db_option(command: argparse.ArgumentParser) -> None:
        command.add_argument("--db", default=".oaa/memory.sqlite3")

    memory_add = memory_actions.add_parser("add", help="save a user-approved memory")
    add_memory_db_option(memory_add)
    memory_add.add_argument("--category", default="fact")
    memory_add.add_argument("--source", default="user")
    memory_add.add_argument("content")

    memory_list = memory_actions.add_parser("list", help="list saved memories")
    add_memory_db_option(memory_list)
    memory_list.add_argument("--category")
    memory_list.add_argument("--limit", type=int, default=100)

    memory_search = memory_actions.add_parser("search", help="search saved memories")
    add_memory_db_option(memory_search)
    memory_search.add_argument("--limit", type=int, default=10)
    memory_search.add_argument("query")

    memory_update = memory_actions.add_parser("update", help="update a saved memory")
    add_memory_db_option(memory_update)
    memory_update.add_argument("memory_id", type=int)
    memory_update.add_argument("content")
    memory_update.add_argument("--category")
    memory_update.add_argument("--source")

    memory_delete = memory_actions.add_parser("delete", help="delete one saved memory")
    add_memory_db_option(memory_delete)
    memory_delete.add_argument("memory_id", type=int)

    memory_clear = memory_actions.add_parser("clear", help="delete all saved memories")
    add_memory_db_option(memory_clear)
    memory_clear.add_argument(
        "--yes", action="store_true",
        help="confirm permanent deletion of all logical memory records",
    )

    return parser


def build_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )


def run_ask(args: argparse.Namespace) -> int:
    engine = Engine()
    engine.load_model(args.model)

    assistant = PersonalAssistant(engine)
    assistant.set_system_prompt(args.system)

    print(assistant.chat(args.prompt, build_config(args)))
    return 0


def run_chat(args: argparse.Namespace) -> int:
    if args.max_history_messages <= 0:
        raise ValueError("max-history-messages must be greater than zero")
    if args.rag_top_k <= 0:
        raise ValueError("rag-top-k must be greater than zero")
    if args.rag_max_context_chars <= 0:
        raise ValueError("rag-max-context-chars must be greater than zero")

    rag_pipeline = None
    if args.docs:
        rag_pipeline = RAGPipeline(
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
        chunk_count = rag_pipeline.ingest_directory(args.docs)
        print(f"Indexed {chunk_count} chunks from {args.docs}")

    engine = Engine()
    engine.load_model(args.model)

    memory_store = MemoryStore(args.memory_db) if args.memory_db else None
    assistant_options = {
        "rag_pipeline": rag_pipeline,
        "rag_top_k": args.rag_top_k,
        "rag_max_context_chars": args.rag_max_context_chars,
    }
    if memory_store is not None:
        assistant_options["memory_store"] = memory_store
        print(f"Persistent memory enabled: {args.memory_db}")
    assistant = PersonalAssistant(engine, **assistant_options)
    assistant.set_system_prompt(args.system)
    assistant.session.max_history_messages = args.max_history_messages

    print("OAA chat. Commands: /reset, /system <text>, /stats, /exit")

    while True:
        try:
            user_input = input("you> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        text = user_input.strip()
        if not text:
            continue

        if text == "/exit":
            return 0

        if text == "/reset":
            assistant.reset()
            print("session reset")
            continue

        if text.startswith("/system "):
            system_prompt = text[8:].strip()
            if not system_prompt:
                print("system prompt cannot be empty")
                continue
            assistant.set_system_prompt(system_prompt)
            print("system prompt updated")
            continue

        if text == "/stats":
            print(assistant.stats())
            continue

        if text.startswith("/system"):
            print("usage: /system <text>")
            continue

        try:
            print("oaa> ", end="", flush=True)
            for chunk in assistant.chat_stream(text, build_config(args)):
                print(chunk, end="", flush=True)
            print()
        except (RuntimeError, ValueError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)


def run_search(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    chunk_count = pipeline.ingest_directory(args.docs)
    if chunk_count == 0:
        print("No extractable document chunks were indexed.")
        return 0

    results = pipeline.search(
        args.query,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    if not results:
        print("No matching chunks found.")
        return 0

    for rank, result in enumerate(results, start=1):
        print(
            f"[{rank}] {result.source} | chunk {result.chunk.chunk_index} "
            f"| similarity {result.score:.3f}"
        )
        print(result.chunk.text)
        print()
    return 0


def run_memory(args: argparse.Namespace) -> int:
    store = MemoryStore(args.db)

    if args.memory_action == "add":
        record = store.add(args.content, category=args.category, source=args.source)
        print(f"Saved memory {record.id} [{record.category}]")
        return 0

    if args.memory_action == "list":
        records = store.list_memories(category=args.category, limit=args.limit)
        if not records:
            print("No saved memories.")
            return 0
        for record in records:
            print(f"[{record.id}] [{record.category}] ({record.source}) {record.content}")
        return 0

    if args.memory_action == "search":
        records = store.search(args.query, limit=args.limit)
        if not records:
            print("No matching memories.")
            return 0
        for record in records:
            print(f"[{record.id}] [{record.category}] ({record.source}) {record.content}")
        return 0

    if args.memory_action == "update":
        record = store.update(
            args.memory_id, args.content, category=args.category, source=args.source
        )
        if record is None:
            print(f"Memory {args.memory_id} not found.", file=sys.stderr)
            return 1
        print(f"Updated memory {record.id} [{record.category}]")
        return 0

    if args.memory_action == "delete":
        if store.delete(args.memory_id):
            print(f"Deleted memory {args.memory_id}.")
            return 0
        print(f"Memory {args.memory_id} not found.", file=sys.stderr)
        return 1

    if args.memory_action == "clear":
        if not args.yes:
            print("Refusing to clear memories without --yes.", file=sys.stderr)
            return 2
        count = store.clear()
        print(f"Deleted {count} memories.")
        return 0

    raise ValueError("unsupported memory action")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "ask":
            return run_ask(args)
        if args.command == "chat":
            return run_chat(args)
        if args.command == "search":
            return run_search(args)
        if args.command == "memory":
            return run_memory(args)
        parser.error("unknown command")
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
