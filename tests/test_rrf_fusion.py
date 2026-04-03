from langchain_core.documents import Document

from agent.retriever import rrf_fuse


def test_rrf_fuse_dedup_and_order():
    d1 = Document(page_content="alpha", metadata={"source": "s1"})
    d2 = Document(page_content="beta", metadata={"source": "s1"})
    d3 = Document(page_content="gamma", metadata={"source": "s2"})

    # list1 prefers d1 then d2
    # list2 prefers d3 then d1
    fused = rrf_fuse([[d1, d2], [d3, d1]], k=60)

    docs = [d for d, _ in fused]

    # should deduplicate d1
    assert len(docs) == 3

    # d1 appears in both lists -> should be ranked first
    assert docs[0].page_content == "alpha"

    # d2 and d3 appear once each; order depends on ranks
    assert set([docs[1].page_content, docs[2].page_content]) == {"beta", "gamma"}
