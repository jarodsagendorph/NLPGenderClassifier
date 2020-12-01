import requests

women = [
    "JANE AUSTEN","CHARLOTTE BRONTE","EMILY BRONTE","CHARLOTTE LENNOX","APHRA BEHN","GEORGE ELIOT",
    "MARY SHELLEY", "MARY WOLLSTONECRAFT", "LOUISA MAY ALCOTT", "ANN RADCLIFFE", "EMILY DICKINSON",
    "LADY GREGORY", "ANNE BRONTE", "ELIZABETH GASKELL", "FRANCES BURNEY", "CLARA REEVE", "MARY DAVYS",
    "ELIZABETH VON ARNIM", "KATE CHOPIN"
]

men = [
    "Charles Dickens",  "Walt Whitman", "Washington Irving", "Edgar Allan Poe", "Herman Melville",
    "Ralph Waldo Emerson", "Henry David Thoreau", "Ida B. Wells", "Jacob Riis", "Margaret Fuller",
    "John Muir", "Frederick Douglass", "Charles Darwin", "Nathaniel Hawthorne", "Horace Greeley",
    "George Perkins Marsh", "Horatio Alger"
]

for w in men:
    r = requests.get("http://gutendex.com/books/", params={"search": w})

    for res in r.json()["results"]:
        if 'en' in res["languages"]:
            formats = [f for f in res["formats"].keys() if f.startswith("text/plain")]
            f = None
            if "text/plain; charset=utf-8" in formats:
                f = "text/plain; charset=utf-8"
            elif "text/plain" in formats:
                f = 'text/plain'
            else:
                f = formats[0]

            print(res["authors"][0]["name"], res["title"], res["languages"], res["formats"][f])
            if not res["formats"][f].endswith('.zip'):
                book = requests.get(res["formats"][f]).text

            with open("books/m/"+res["authors"][0]["name"]+"||"+res["title"], "w") as f:
                f.write(book)

