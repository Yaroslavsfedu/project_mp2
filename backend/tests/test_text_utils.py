from app.ml_pipeline.text_utils import (
    chunk_text,
    cosine,
    expand_query,
    hashing_embedding,
    keywords,
    make_snippet,
    normalize_text,
    split_sentences,
)


def test_normalize_text_joins_hyphenated_words():
    raw = "Министер-\nство науки и высшего\n\n\nобразования"
    cleaned = normalize_text(raw)
    assert "Министерство" in cleaned
    assert "\n\n\n" not in cleaned


def test_normalize_text_fixes_digit_letter_confusions():
    assert "0" in normalize_text("ИНН 61О5О48312")


def test_split_sentences():
    sentences = split_sentences("Первое предложение. Второе предложение! Третье?")
    assert len(sentences) == 3


def test_keywords_excludes_stopwords():
    text = "Договор поставки оборудования. Поставка оборудования осуществляется по договору."
    result = keywords(text, limit=4)
    assert "поставки" in result or "поставка" in result
    assert "и" not in result


def test_chunk_text_respects_limit():
    text = "\n\n".join(["Абзац " + "слово " * 40 for _ in range(6)])
    chunks = chunk_text(text, max_chars=500)
    assert len(chunks) > 1
    assert all(len(chunk) <= 520 for chunk in chunks)


def test_hashing_embedding_similarity():
    a = hashing_embedding("договор поставки вычислительной техники", 512)
    b = hashing_embedding("поставка компьютерной техники по договору", 512)
    c = hashing_embedding("погода в ростове на дону сегодня", 512)
    assert len(a) == 512
    assert cosine(a, b) > cosine(a, c)


def test_expand_query_synonyms():
    terms = expand_query("штраф за просрочку")
    assert "неустойка" in terms or "пеня" in terms


def test_make_snippet_marks_match():
    text = "Преамбула. " * 20 + "Неустойка составляет 0,1% за каждый день просрочки."
    snippet, found = make_snippet(text, ["неустойка"])
    assert found is True
    assert "Неустойка" in snippet
    assert len(snippet) < len(text)
