import xml.etree.ElementTree as ET
from typing import List, Dict
from dataclasses import dataclass
import tiktoken

# Token encoder for OpenAI-like models
token_encoder = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(token_encoder.encode(text))

@dataclass
class Node:
    id: str
    text: str
    children: List['Node']
    metadata: Dict[str, str]


def normalize_richcontent(rc_elem: ET.Element) -> str:
    """
    Extracts and flattens richcontent (notes) into a simple string.
    Links become [Link: URL], images become [Image: path], text is preserved.
    """
    parts: List[str] = []
    # hyperlinks
    for a in rc_elem.findall('.//a'):
        href = a.get('href')
        if href:
            parts.append(f"[Link: {href}]")
    # images
    for img in rc_elem.findall('.//img'):
        src = img.get('src')
        if src:
            parts.append(f"[Image: {src}]")
    # plaintext inside richcontent
    for txt in rc_elem.itertext():
        t = txt.strip()
        if t and not t.startswith(('http://', 'https://', '[Link:', '[Image:')):
            parts.append(t)
    return ' '.join(parts)


def parse_node(elem: ET.Element) -> Node:
    node_id = elem.get('ID', '')
    text = elem.get('TEXT', '')
    # capture standard attributes
    metadata = {k: v for k, v in elem.items() if k not in ('ID', 'TEXT')}
    # capture richcontent notes
    notes = []
    for rc in elem.findall('richcontent'):
        if rc.get('TYPE', '').upper() == 'NOTE':
            notes.append(normalize_richcontent(rc))
    if notes:
        metadata['notes'] = ' '.join(notes)
    # recurse into child nodes
    children = [parse_node(child) for child in elem.findall('node')]
    return Node(id=node_id, text=text, children=children, metadata=metadata)


def parse_mm(filepath: str) -> Node:
    tree = ET.parse(filepath)
    root_elem = tree.getroot().find('node')
    if root_elem is None:
        raise ValueError("No <node> element found in MM file")
    return parse_node(root_elem)


def node_to_markdown(node: Node, depth: int = 1) -> str:
    lines: List[str] = []
    # Heading reflects depth
    prefix = '#' * depth
    heading = f"{prefix} {node.text}" if node.text else prefix
    lines.append(heading)
    # metadata comment block
    if node.metadata:
        meta = '; '.join(f"{k}={v}" for k, v in node.metadata.items())
        lines.append(f"<!-- {meta} -->")
    # inline note rendering
    note = node.metadata.get('notes', '')
    if note:
        lines.append(f"> Note: {note}")
    # child subtrees
    for child in node.children:
        lines.append(node_to_markdown(child, depth + 1))
    return '\n'.join(lines)

def chunk_node(node: Node, max_tokens: int = 1000) -> List[str]:
    md = node_to_markdown(node)
    if count_tokens(md) <= max_tokens:
        return [md]

    chunks: List[str] = []

    if not node.children:
        # Leaf node with large content: token-aware splitting
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(md)
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i+max_tokens]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text.strip())
        
        return chunks

    # Internal node: recurse into children
    for child in node.children:
        chunks.extend(chunk_node(child, max_tokens))

    return chunks




if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python freeplane_parser.py path/to/mindmap.mm [max_tokens]")
        sys.exit(1)
    filepath = sys.argv[1]
    max_toks = int(sys.argv[2]) if len(sys.argv) >= 3 else 1000
    root = parse_mm(filepath)
    chunks = chunk_node(root, max_toks)
    for i, c in enumerate(chunks, 1):
        print(f"--- Chunk {i} ({count_tokens(c)} tokens) ---")
        print(c)
        print()
