"""Read published BKS and HALNS values from the supplied XLSX files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any
import xml.etree.ElementTree as ET
from zipfile import ZipFile

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True, slots=True)
class PublishedReference:
    instance: str
    best_known: float | None
    halns_score: float | None
    halns_cpu_seconds: float | None = None
    dataset: str = ""


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    if letters is None:
        return 0
    value = 0
    for letter in letters.group():
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def _xlsx_rows(path: Path) -> dict[str, list[list[Any]]]:
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{MAIN_NS}}}si"):
                shared_strings.append(
                    "".join(
                        text.text or ""
                        for text in item.iter(f"{{{MAIN_NS}}}t")
                    )
                )

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships.findall(
                f"{{{PKG_REL_NS}}}Relationship"
            )
        }
        result: dict[str, list[list[Any]]] = {}
        for sheet in workbook.findall(
            f".//{{{MAIN_NS}}}sheet"
        ):
            name = sheet.attrib["name"]
            relation_id = sheet.attrib[f"{{{REL_NS}}}id"]
            target = targets[relation_id].lstrip("/")
            if not target.startswith("xl/"):
                target = str(PurePosixPath("xl") / target)
            worksheet = ET.fromstring(archive.read(target))
            rows: list[list[Any]] = []
            for row in worksheet.findall(
                f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row"
            ):
                values: dict[int, Any] = {}
                for cell in row.findall(f"{{{MAIN_NS}}}c"):
                    index = _column_index(cell.attrib.get("r", "A1"))
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find(f"{{{MAIN_NS}}}v")
                    if cell_type == "inlineStr":
                        value: Any = "".join(
                            text.text or ""
                            for text in cell.iter(f"{{{MAIN_NS}}}t")
                        )
                    elif value_node is None:
                        value = None
                    elif cell_type == "s":
                        value = shared_strings[int(value_node.text or "0")]
                    else:
                        raw = value_node.text or ""
                        try:
                            number = float(raw)
                            value = (
                                int(number)
                                if number.is_integer()
                                else number
                            )
                        except ValueError:
                            value = raw
                    values[index] = value
                if values:
                    width = max(values) + 1
                    rows.append([values.get(index) for index in range(width)])
            result[name] = rows
        return result


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_published_references(
    benchmark_root: str | Path,
) -> dict[str, PublishedReference]:
    """Load references keyed by lowercase instance stem."""
    root = Path(benchmark_root)
    references: dict[str, PublishedReference] = {}

    chao_file = (
        root / "Chao et al., (1996)" / "Solutions-Published.xlsx"
    )
    if chao_file.exists():
        for rows in _xlsx_rows(chao_file).values():
            if not rows:
                continue
            header = [str(value or "").strip() for value in rows[0]]
            try:
                instance_col = header.index("Instance")
                best_col = header.index("Best")
                halns_col = header.index("HALNS")
            except ValueError:
                continue
            for row in rows[1:]:
                if instance_col >= len(row) or not row[instance_col]:
                    continue
                instance_name = str(row[instance_col]).strip()
                references[instance_name.lower()] = PublishedReference(
                    instance=instance_name,
                    best_known=(
                        _number(row[best_col])
                        if best_col < len(row)
                        else None
                    ),
                    halns_score=(
                        _number(row[halns_col])
                        if halns_col < len(row)
                        else None
                    ),
                    dataset="Chao1996",
                )

    dang_file = (
        root / "Dang et al., (2013)" / "Solutions - Published.xlsx"
    )
    if dang_file.exists():
        rows_by_sheet = _xlsx_rows(dang_file)
        for rows in rows_by_sheet.values():
            for row in rows[2:]:
                if not row or not row[0]:
                    continue
                instance_name = str(row[0]).strip()
                if instance_name.lower().startswith("number of"):
                    continue
                references[instance_name.lower()] = PublishedReference(
                    instance=instance_name,
                    best_known=_number(row[1] if len(row) > 1 else None),
                    halns_score=_number(
                        row[7] if len(row) > 7 else None
                    ),
                    halns_cpu_seconds=_number(
                        row[8] if len(row) > 8 else None
                    ),
                    dataset="Dang2013",
                )
    return references
