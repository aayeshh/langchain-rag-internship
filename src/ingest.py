"""
Ingestion pipeline: load contracts (PDF + DOCX) from data/contracts/,
split into chunks, embed, and persist to a Chroma vector store.

Re-embedding is skipped for documents that haven't changed since the
last run (tracked via a content-hash manifest next to the vector store) --
see `_load_manifest` / `_files_to_embed` below. This is what the brief's
"caching: avoid re-embedding unchanged documents" requirement maps to;
`set_llm_cache` (config in src/config.py, used from src/agent.py) is the
separate LLM-call cache for repeated queries.

Run directly: python -m src.ingest
"""

import hashlib
import json
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, CONTRACTS_DIR

MANIFEST_PATH = CHROMA_DIR / "ingest_manifest.json"

LOADERS_BY_SUFFIX = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def discover_contract_files(contracts_dir: Path = CONTRACTS_DIR) -> list[Path]:
    """All PDF/DOCX files under contracts_dir, sorted for deterministic order."""
    files = [
        p
        for p in contracts_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in LOADERS_BY_SUFFIX
    ]
    return sorted(files)


def files_to_embed(files: list[Path], manifest: dict) -> list[Path]:
    """Return only the files whose content hash differs from what's in the
    manifest (new files or edited files) -- this is the re-embedding guard."""
    changed = []
    for f in files:
        current_hash = _file_hash(f)
        if manifest.get(str(f)) != current_hash:
            changed.append(f)
    return changed


def load_documents(files: list[Path]) -> list[Document]:
    """Load each file with the right loader for its extension. Skips a file
    (with a warning) instead of crashing the whole ingest run if one file is
    corrupt/unreadable -- required by the brief's robustness criterion."""
    docs: list[Document] = []
    for f in files:
        loader_cls = LOADERS_BY_SUFFIX[f.suffix.lower()]
        try:
            loaded = loader_cls(str(f)).load()
        except Exception as e:
            print(f"  [WARN] skipping {f.name}: failed to load ({e})")
            continue
        if not loaded or not any(d.page_content.strip() for d in loaded):
            print(f"  [WARN] {f.name} loaded but produced no extractable text "
                  f"(likely a scanned/image-only PDF) -- skipping")
            continue
        # Normalize metadata: always have a clean "source" filename, since
        # PyPDFLoader gives a full path and we want short, citable names.
        for d in loaded:
            d.metadata["source"] = f.name
        docs.extend(loaded)
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks: list[Document]):
    """Embed and persist new chunks into the Chroma store. Imports the
    embeddings/Chroma dependencies lazily so this module can still be used
    for load/split testing without OPENAI_API_KEY set."""
    from langchain_chroma import Chroma

    from src.config import get_embeddings

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )
    if chunks:
        vectorstore.add_documents(chunks)
    return vectorstore


def ingest(contracts_dir: Path = CONTRACTS_DIR, embed: bool = True):
    """Full pipeline: discover -> skip unchanged -> load -> split -> embed.

    embed=False lets you exercise load/split logic without needing
    OPENAI_API_KEY (useful for quick local testing).
    """
    print(f"Scanning {contracts_dir} ...")
    all_files = discover_contract_files(contracts_dir)
    print(f"  found {len(all_files)} contract file(s)")

    manifest = _load_manifest()
    changed_files = files_to_embed(all_files, manifest)
    unchanged_count = len(all_files) - len(changed_files)
    print(f"  {unchanged_count} unchanged (skipped), {len(changed_files)} new/changed")

    if not changed_files:
        print("Nothing new to ingest.")
        return None, []

    print("Loading changed documents...")
    docs = load_documents(changed_files)
    print(f"  loaded {len(docs)} page(s)/doc(s)")

    if embed:
        print("Extracting structured summaries...")
        docs_by_source: dict[str, list] = {}
        for d in docs:
            docs_by_source.setdefault(d.metadata["source"], []).append(d)
        if docs_by_source:
            from src.extraction import extract_and_cache_for_files

            summaries = extract_and_cache_for_files(docs_by_source)
            print(f"  extracted {len(summaries)} summar{'y' if len(summaries) == 1 else 'ies'}")
    else:
        print("  embed=False: skipped structured extraction (also needs the API)")

    print("Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"  {len(docs)} page(s) -> {len(chunks)} chunk(s)")

    vectorstore = None
    if embed:
        print("Embedding and persisting to Chroma...")
        vectorstore = build_vectorstore(chunks)
        # Only mark files as done once they're actually embedded.
        for f in changed_files:
            manifest[str(f)] = _file_hash(f)
        _save_manifest(manifest)
        print(f"  persisted to {CHROMA_DIR}")
    else:
        print("  embed=False: skipped embedding step (load/split test only)")

    return vectorstore, chunks


if __name__ == "__main__":
    ingest()
