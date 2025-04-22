import os
import pytest
from second_brain_chat.index import index_chunks, search_chunks
from second_brain_chat.freeplane_parser import parse_mm, chunk_node
from langchain.schema import Document
import tempfile

# --- Fixtures ---

@pytest.fixture
def basic_mindmap():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="Child A">
                <node TEXT="Grandchild A1"/>
            </node>
            <node TEXT="Child B"/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()
        yield tmp.name
    os.remove(tmp.name)

@pytest.fixture
def indexed_chunks(basic_mindmap):
    root = parse_mm(basic_mindmap)
    chunks = chunk_node(root, max_tokens=200)
    vectorstore = index_chunks(chunks, metadata={"source": "test.mm"})
    return vectorstore, chunks

# --- Tests ---

def test_query_returns_expected_chunk(indexed_chunks):
    store, chunks = indexed_chunks
    results = search_chunks("Grandchild A1", store, top_k=3)
    assert results, "No results returned for known node"
    joined = "\n".join(r.page_content for r in results)
    assert "Grandchild A1" in joined
    assert "Child A" in joined, "Parent context missing"

def test_chunk_metadata_preserved(indexed_chunks):
    store, _ = indexed_chunks
    results = search_chunks("Child B", store)
    assert results
    for doc in results:
        assert isinstance(doc, Document)
        assert "source" in doc.metadata
        assert doc.metadata["source"] == "test.mm"

def test_retrieval_fidelity_multilingual():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="こんにちは"/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False, encoding="utf-8") as tmp:
        tmp.write(mm_xml)
        tmp.flush()
        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)
        store = index_chunks(chunks)
        results = search_chunks("こんにちは", store)
        assert any("こんにちは" in r.page_content for r in results)

def test_partial_match_context_expansion(indexed_chunks):
    store, _ = indexed_chunks
    results = search_chunks("A1", store)
    assert any("Child A" in r.page_content for r in results), "Expected parent context not found"

def test_deduplication_on_reindex():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="Alpha"/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()

        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)
        store = index_chunks(chunks)
        store = index_chunks(chunks, existing_vectorstore=store)

        results = search_chunks("Alpha", store)
        texts = [r.page_content for r in results]
        assert len(set(texts)) == len(texts), "Duplicates found after re-indexing"

def test_search_result_ranking():
    mm_xml = '''<?xml version="1.0"?>
    <map version="1.0.1">
        <node TEXT="Root">
            <node TEXT="Exact match"/>
            <node TEXT="Partial match example"/>
        </node>
    </map>'''
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mm", delete=False) as tmp:
        tmp.write(mm_xml)
        tmp.flush()
        root = parse_mm(tmp.name)
        chunks = chunk_node(root, max_tokens=200)
        store = index_chunks(chunks)
        results = search_chunks("Exact match", store, top_k=2)

        assert results
        assert "Exact match" in results[0].page_content

def test_empty_query_returns_nothing(indexed_chunks):
    store, _ = indexed_chunks
    results = search_chunks("", store)
    assert results == [] or all(len(r.page_content.strip()) == 0 for r in results)
