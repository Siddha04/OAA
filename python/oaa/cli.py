from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .assistant import PersonalAssistant
from .config import GenerationConfig
from .engine import Engine
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

    assistant = PersonalAssistant(
        engine,
        rag_pipeline=rag_pipeline,
        rag_top_k=args.rag_top_k,
        rag_max_context_chars=args.rag_max_context_chars,
    )
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
        parser.error("unknown command")
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
