import os
import re
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict
from dataclasses import dataclass
import tiktoken

# Configuration: allow overriding tokenizer encoding via env or CLI
ENCODING_NAME = os.getenv('TOKEN_ENCODING', 'cl100k_base')

def _get_encoder(name: str):
    try:
        return tiktoken.get_encoding(name)
    except Exception as e:
        raise ValueError(f"Unable to load encoding '{name}': {e}")

# Helper: count tokens using OpenAI encoding
def count_tokens(text: str, encoding_name: str = None) -> int:
    name = encoding_name or ENCODING_NAME
    enc = _get_encoder(name)
    return len(enc.encode(text))

# Whitelist of Freeplane attributes to include as metadata
ALLOWED_METADATA = {'CREATED', 'LINK', 'FOLDED', 'STYLE', 'POSITION', 'MODIFIED'}

@dataclass
class Node:
    id: str
    text: str
    children: List['Node']
    metadata: Dict[str, str]


def parse_node(elem: ET.Element) -> Node:
    try:
        node_id = elem.get('ID', '')
        text = elem.get('TEXT', '')
        # Sanitize Markdown special characters in text
        text = escape_markdown(text)
        # Capture a whitelist of attributes
        metadata = {k: v for k, v in elem.items() if k in ALLOWED_METADATA}
        children = [parse_node(child) for child in elem.findall('node')]
        return Node(id=node_id, text=text, children=children, metadata=metadata)
    except RecursionError:
        raise ValueError(f"Node recursion too deep at element with ID={elem.get('ID')}")
    except Exception as e:
        raise ValueError(f"Error parsing node ID={elem.get('ID')}: {e}")


def parse_mm(filepath: str) -> Node:
    """
    Load a Freeplane .mm file and return the root Node.
    """
    try:
        tree = ET.parse(filepath)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file '{filepath}': {e}")
    root_elem = tree.getroot().find('node')
    if root_elem is None:
        raise ValueError(f"No <node> element found in MM file '{filepath}'")
    return parse_node(root_elem)


def escape_markdown(text: str) -> str:
    """
    Escape Markdown-sensitive characters in the given text.
    """
    # characters: \ ` * _ { } [ ] ( ) # + - . !
    return re.sub(r'([\\`*_\{\}\[\]\(\)#\+\-\.!])', r"\\\1", text)


def node_to_markdown(node: Node, depth: int = 1) -> str:
    """
    Recursively convert a Node tree into Markdown, using headings for hierarchy.
    """
    lines: List[str] = []
    prefix = '#' * depth
    # Heading line
    lines.append(f"{prefix} {node.text}" if node.text else f"{prefix}")
    # Metadata as comment
    if node.metadata:
        meta = '; '.join(f"{k}={v}" for k, v in node.metadata.items())
        lines.append(f"<!-- {meta} -->")
    # Recurse
    for child in node.children:
        lines.append(node_to_markdown(child, depth + 1))
    return '\n'.join(lines)


def chunk_node(node: Node, max_tokens: int = 1000) -> List[str]:
    """
    Tree-aware chunking: if a node's full subtree fits under max_tokens, emit it as one chunk;
    otherwise, recursively chunk its children.
    """
    md = node_to_markdown(node)
    if count_tokens(md) <= max_tokens:
        return [md]
    # Otherwise dive into children
    chunks: List[str] = []
    for child in node.children:
        chunks.extend(chunk_node(child, max_tokens))
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Freeplane .mm to Markdown chunker")
    parser.add_argument('filepath', help='Path to the Freeplane .mm file')
    parser.add_argument('--max_tokens', type=int, default=1000,
                        help='Maximum tokens per chunk')
    parser.add_argument('--encoding', type=str,
                        help='Optional tokenizer encoding name')
    args = parser.parse_args()

    if args.encoding:
        global ENCODING_NAME
        ENCODING_NAME = args.encoding

    try:
        root = parse_mm(args.filepath)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    chunks = chunk_node(root, args.max_tokens)
    for i, chunk in enumerate(chunks, 1):
        tok_count = count_tokens(chunk)
        print(f"--- Chunk {i} ({tok_count} tokens) ---")
        print(chunk)
        print()

if __name__ == '__main__':
    main()
