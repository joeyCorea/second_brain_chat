import pytest
from second_brain_chat.freeplane_parser import parse_mm, chunk_node, count_tokens
import tempfile

# Utility to generate a large Freeplane XML string with N children under root
def generate_large_mm(n=20):
    children = "".join(
        f'<node TEXT="Child {i}" POSITION="right"><node TEXT="Grandchild {i}-1"/><node TEXT="Grandchild {i}-2"/></node>'
        for i in range(n)
    )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
    <map version="1.0.1">
        <node TEXT="Root Node" CREATED="1710000000000">
            {children}
        </node>
    </map>'''

def test_chunking_large_tree():
    mm_xml = generate_large_mm(n=30)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()
        
        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)

        assert len(chunks) > 1

        for c in chunks:
            tok_count = count_tokens(c)
            assert tok_count <= 200, f"Chunk too large: {tok_count} tokens"

        for c in chunks:
            assert "#" in c, "Chunk missing heading"
            assert len(c.strip()) > 0

        for i, c in enumerate(chunks):
            print(f"Chunk {i+1}: {count_tokens(c)} tokens\n{c.splitlines()[0]}\n")

def test_empty_text_node():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()

        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)

        assert any(line.startswith("#") for c in chunks for line in c.splitlines()), "Expected headings in output"

def test_duplicate_text_nodes():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="Repeat"/>
            <node TEXT="Repeat"/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()

        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)

        matches = [line for c in chunks for line in c.splitlines() if "Repeat" in line]
        assert len(matches) == 2, "Expected both Repeat nodes to appear"

def test_mixed_metadata():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root" CREATED="123456" MODIFIED="654321">
            <node TEXT="Child"/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()

        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)

        metadata_lines = [line for c in chunks for line in c.splitlines() if line.startswith("<!--")]
        # we want the metadata to appear on one line each for easy grepability and readaibility
        created = any("CREATED=" in line for line in metadata_lines)
        modified = any("MODIFIED=" in line for line in metadata_lines)
        assert created and modified, "Expected CREATED and MODIFIED comments"

def test_all_nodes_accounted_for():
    mm_xml = generate_large_mm(n=25)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()

        root = parse_mm(tmp.name)

        def count_nodes(n):
            return 1 + sum(count_nodes(c) for c in n.children)
        total_nodes = count_nodes(root)

        chunks = chunk_node(root, max_tokens=200)
        headings = [line for c in chunks for line in c.splitlines() if line.startswith("#")]

        assert len(headings) / total_nodes >= 0.9, f"Coverage below 90% ({len(headings)} of {total_nodes})"

def test_nodes_with_links_and_images():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="See docs">
                <richcontent TYPE="NOTE"><html><body><a href="https://example.com">Link</a></body></html></richcontent>
            </node>
            <node TEXT="Visual">
                <richcontent TYPE="NOTE"><html><body><img src="image.png"/></body></html></richcontent>
            </node>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()

        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)

        output = "\n".join(chunks)
        assert "https://example.com" in output or "[Link]" in output, "Link not represented"
        assert "image.png" in output or "[Image]" in output, "Image ref not surfaced"
