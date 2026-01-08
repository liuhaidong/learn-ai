import openpyxl
from typing import List


def parse_excel(path: str) -> List:
    wb = openpyxl.load_workbook(path)
    sheet = wb.active
    rows = [list(r) for r in sheet.iter_rows(values_only=True)]
    return rows
