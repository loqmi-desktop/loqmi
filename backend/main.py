"""
backend/main.py — Dev CLI entry point
Creates a single persistent chat session for testing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core import memory, model_manager, router

DEV_CHAT_ID = "dev-cli-session"


def main():
    memory.init_db()

    # Create dev chat if it doesn't exist yet
    existing = [c["id"] for c in memory.list_chats()]
    if DEV_CHAT_ID not in existing:
        memory.create_chat(DEV_CHAT_ID)

    print("loqmi dev CLI.")
    print("Commands: 'quit', 'reset' (clear history), 'rag on', 'rag off'\n")

    rag_state = memory.get_rag_enabled(DEV_CHAT_ID)
    print(f"RAG is {'ON' if rag_state else 'OFF'}\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except KeyboardInterrupt:
            break

        if not prompt:
            continue

        if prompt.lower() == "quit":
            break

        if prompt.lower() == "reset":
            memory.clear_history(DEV_CHAT_ID)
            print("[memory cleared]\n")
            continue

        if prompt.lower() == "rag on":
            memory.set_rag_enabled(DEV_CHAT_ID, True)
            print("[RAG enabled]\n")
            continue

        if prompt.lower() == "rag off":
            memory.set_rag_enabled(DEV_CHAT_ID, False)
            print("[RAG disabled]\n")
            continue

        print("Assistant: ", end="", flush=True)
        for chunk in router.handle(prompt, DEV_CHAT_ID):
            if chunk["type"] == "token":
                print(chunk["text"], end="", flush=True)
            elif chunk["type"] == "stats":
                print(f"\n[{chunk['category']} | classify: {chunk['classify_ms']}ms | inference: {chunk['inference_s']}s]\n")

    model_manager.unload()
    print("Exiting.")


if __name__ == "__main__":
    main()