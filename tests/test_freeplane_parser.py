import pytest
from second_brain_chat.freeplane_parser import parse_mm, chunk_node, count_tokens, normalize_richcontent
import tempfile
import xml.etree.ElementTree as ET

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

def test_normalize_richcontent():
    rc = ET.fromstring('''<richcontent TYPE="NOTE">
        <html>
            <head/><body>
                Some <b>bold</b> text, a <a href="https://example.com">link</a>, and an image:
                <img src="image.png"/>
            </body>
        </html>
    </richcontent>''')
    normalized = normalize_richcontent(rc)
    assert "Some" in normalized
    assert "bold" in normalized
    assert "[Link: https://example.com]" in normalized
    assert "[Image: image.png]" in normalized

def test_preserves_sibling_order():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="First"/>
            <node TEXT="Second"/>
            <node TEXT="Third"/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()
        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)
        flat_output = "\n".join(chunks)

        first_idx = flat_output.find("First")
        second_idx = flat_output.find("Second")
        third_idx = flat_output.find("Third")

        assert first_idx < second_idx < third_idx, "Sibling order not preserved"

def test_single_node_exceeds_token_limit():
    large_text = "Word " * 1000  # ~1000 tokens
    mm_xml = f'''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="{large_text.strip()}"/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()

        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)

        assert len(chunks) > 0, "No chunks returned for large content"
        token_counts = [count_tokens(c) for c in chunks]
        assert all(tc <= 200 for tc in token_counts), f"At least one chunk too large: {token_counts}"
        joined = "\n".join(chunks)
        assert "Word" in joined, "Expected large content to be chunked, but it’s missing"

def test_malformed_xml_raises():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write("<map><node TEXT='Missing end'")
        tmp.flush()
        with pytest.raises(Exception):
            parse_mm(tmp.name)

def test_markdown_depth_levels():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="Child">
                <node TEXT="Grandchild"/>
            </node>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()
        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)
        flat_output = "\n".join(chunks)
        assert "# Root" in flat_output
        assert "## Child" in flat_output
        assert "### Grandchild" in flat_output

def test_normalize_richcontent_with_junk_html():
    rc = ET.fromstring('''<richcontent TYPE="NOTE">
        <html><body><div><unknown><a>Broken</a></unknown></div></body></html>
    </richcontent>''')
    out = normalize_richcontent(rc)
    assert "Broken" in out

def test_preserves_text_exactly():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root &amp; Co. — test &quot;quotes&quot; and symbols ☂️"/>    
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False, encoding="utf-8") as tmp:
        tmp.write(mm_xml)
        tmp.flush()
        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)
        joined = "\n".join(chunks)
        print("joined:\n")
        print(joined)
        assert "Root & Co." in joined
        assert "☂️" in joined
        assert "\"quotes\"" in joined or '\"quotes\"' in joined

@pytest.mark.parametrize("text", [
    "🙂 🚀 🔥",                      # emojis
    "`code_block` with symbols ~!@",  # code + punctuation
    "नमस्ते दुनिया",                 # Hindi
    "漢字とかな",                   # Japanese
])
def test_tokenizer_handles_unicode(text):
    tok_count = count_tokens(text)
    assert tok_count > 0

def test_richcontent_missing_and_empty_body():
    # Case 1: Well-formed, but <body> is missing entirely
    missing_body = ET.fromstring('<richcontent TYPE="NOTE"><html><head></head></html></richcontent>')
    assert normalize_richcontent(missing_body) == "", "Expected empty string when <body> is missing"
    # Case 2: Well-formed with an empty <body>
    empty_body = ET.fromstring('<richcontent TYPE="NOTE"><html><body></body></html></richcontent>')
    assert normalize_richcontent(empty_body) == "", "Expected empty string when <body> has no content"
    # Case 3: Normal case with content in <body>
    has_content = ET.fromstring('<richcontent TYPE="NOTE"><html><body><p>Hi</p></body></html></richcontent>')
    out = normalize_richcontent(has_content)
    assert "Hi" in out, "Expected text content from <body> to be included"

def test_chunk_uniqueness():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="Child 1"/>
            <node TEXT="Child 2"/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()

        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)

        # Collect all text content in chunks
        chunk_texts = [line for c in chunks for line in c.splitlines()]

        # Ensure that no text appears in multiple chunks
        chunk_text_set = set(chunk_texts)
        assert len(chunk_texts) == len(chunk_text_set), "Duplicate node content found in chunks"

#TODO: feels like this already exists
def test_token_limit_enforcement():
    # Construct the XML string with long text
    long_text = " ".join(["longtext"]*50)  # This will generate a long string of 50 "longtext" words

    mm_xml = f'''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="Child 1 with a lot of text to fill up the token limit. {long_text}"/>
            <node TEXT="Child 2"/>
        </node>
    </map>'''
    
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False, encoding="utf-8") as tmp:
        tmp.write(mm_xml)
        tmp.flush()

        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)

        # Ensure no chunk exceeds the max_tokens
        for c in chunks:
            tok_count = count_tokens(c)
            assert tok_count <= 200, f"Chunk exceeds token limit: {tok_count}"

def test_all_nodes_accounted_for():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="Child 1"/>
            <node TEXT="Child 2"/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()

        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)

        # Flatten chunks into headings
        chunk_headings = [line for c in chunks for line in c.splitlines() if line.startswith("#")]

        # Total nodes in the original structure
        def count_nodes(n):
            return 1 + sum(count_nodes(c) for c in n.children)

        total_nodes = count_nodes(root)

        # Ensure at least 90% of the nodes have corresponding headings
        assert len(chunk_headings) / total_nodes >= 0.9, f"Coverage below 90% ({len(chunk_headings)} of {total_nodes})"

def test_heading_disambiguation():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="Child 1"/>
            <node TEXT="Child 2"/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()

        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)

        # Check that all chunks have disambiguated headings
        disambiguated_headings = [
            line for c in chunks for line in c.splitlines() if line.startswith("#") and "Root" in line
        ]
        print(disambiguated_headings)
        assert all("Root" in heading for heading in disambiguated_headings), "Headings are not disambiguated properly"
