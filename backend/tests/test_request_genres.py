from moviematch.llm.request_genres import infer_request_genres

VOCAB = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]


def test_romantic_request_maps_to_romance():
    assert infer_request_genres("romantic movie", VOCAB) == ["Romance"]
    assert infer_request_genres("I want a love story", VOCAB) == ["Romance"]


def test_multiple_genres_keep_request_order():
    assert infer_request_genres("a rom-com for date night", VOCAB) == ["Romance", "Comedy"]
    assert infer_request_genres("science fiction thriller", VOCAB) == ["Sci-Fi", "Thriller"]


def test_negated_genres_are_ignored():
    assert infer_request_genres("romantic but not too scary", VOCAB) == ["Romance"]
    assert infer_request_genres("something like Interstellar, with less action", VOCAB) == []


def test_empty_or_unknown_requests_give_no_genres():
    assert infer_request_genres("", VOCAB) == []
    assert infer_request_genres("something good", VOCAB) == []


def test_genres_missing_from_the_vocabulary_are_dropped():
    assert infer_request_genres("romantic comedy", ["Comedy"]) == ["Comedy"]
