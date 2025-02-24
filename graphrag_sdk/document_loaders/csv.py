import csv
from typing import Iterator
from graphrag_sdk.document import Document


class CSVLoader:
    """
    CSV loader
    """

    def __init__(self, path: str, rows_per_document: int = 500):
        self.path = path
        self.rows_per_document = rows_per_document

    def load(self) -> Iterator[Document]:
        with open(self.path, "r") as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
            num_rows = len(rows)
            num_documents = num_rows // self.rows_per_document
            for i in range(num_documents):
                content = "\n".join(
                    [
                        ",".join(row)
                        for row in rows[
                            i
                            * self.rows_per_document : (i + 1)
                            * self.rows_per_document
                        ]
                    ]
                )
                yield Document(content, f"{self.path}#{i}")
