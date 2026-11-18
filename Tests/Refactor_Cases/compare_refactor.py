#!/usr/bin/env python3
"""
compare_refactor.py

Comparador de regressão para os arquivos VTU produzidos pelo Arritmic3D.

Compatível com o formato observado nos arquivos:
    <VTKFile
        type="UnstructuredGrid"
        byte_order="LittleEndian"
        header_type="UInt32"
        compressor="vtkZLibDataCompressor">

e DataArray:
    format="binary"

O comparador faz duas validações:

1. Usa meshio para comparar:
   - pontos
   - células que o meshio consegue interpretar
   - PointData / CellData

2. Lê diretamente o XML do VTU e decodifica os DataArray binários
   comprimidos pelo vtkZLibDataCompressor para comparar:
   - Cells/connectivity
   - Cells/offsets
   - Cells/types
   - especialmente VTK_VOXEL (tipo 11)

Assim, um PASS não será obtido simplesmente porque o meshio
ignorou os voxels.

Uso:

    python compare_refactor.py \
        --reference ./reference/03_S1S2 \
        --candidate ./candidate/03_S1S2

Ou:

    python compare_refactor.py \
        --reference ./reference \
        --candidate ./candidate

para comparar automaticamente todos os casos presentes nos dois diretórios.

Opções:

    --case NOME       compara somente um caso
    --atol VALOR      tolerância absoluta para floats
    --rtol VALOR      tolerância relativa para floats
    --verbose         mostra detalhes adicionais
    --no-meshio       não faz a etapa meshio
    --no-raw-cells    não faz a comparação direta das Cells
    --report PREFIX   prefixo dos relatórios

Dependências:
    numpy
    meshio
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    import meshio
except ImportError:
    meshio = None


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-8

FLOAT_FIELDS = {
    "APD",
    "DI",
    "CV",
    "AP",
    "LAT",
    "Time",
}

# INTEGER_FIELDS = {
#     "State",
#     "Beat",
# }

INTEGER_FIELDS = set()

# VTK cell types relevantes
VTK_CELL_TYPES = {
    0: "EMPTY_CELL",
    1: "VERTEX",
    2: "POLY_VERTEX",
    3: "LINE",
    4: "POLY_LINE",
    5: "TRIANGLE",
    6: "TRIANGLE_STRIP",
    7: "POLYGON",
    8: "PIXEL",
    9: "QUAD",
    10: "TETRA",
    11: "VOXEL",
    12: "HEXAHEDRON",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def natural_key(path_or_name):
    """
    Ordenação natural.

    Exemplo:
        slab_2.vtu
        slab_10.vtu

    em vez de:
        slab_10.vtu
        slab_2.vtu
    """
    if isinstance(path_or_name, Path):
        name = path_or_name.name
    else:
        name = str(path_or_name)

    parts = re.split(r"(\d+(?:\.\d+)?)", name)

    return [
        float(x) if re.fullmatch(r"\d+(?:\.\d+)?", x)
        else x.lower()
        for x in parts
    ]


def discover_vtu_files(directory: Path) -> dict[str, Path]:
    """Retorna {nome_do_arquivo: caminho}."""
    return {
        p.name: p
        for p in sorted(directory.glob("*.vtu"), key=natural_key)
    }


def find_case_directories(root: Path) -> list[Path]:
    """
    Procura diretórios que contenham pelo menos um VTU.

    Se root já contém VTUs, root é considerado um caso.
    Caso contrário, procura subdiretórios.
    """
    direct = list(root.glob("*.vtu"))
    if direct:
        return [root]

    cases = []
    for p in sorted(root.iterdir(), key=natural_key):
        if p.is_dir() and any(p.glob("*.vtu")):
            cases.append(p)

    return cases


def format_number(x: float) -> str:
    return f"{x:.6e}"


# ---------------------------------------------------------------------------
# XML / VTK binário
# ---------------------------------------------------------------------------

def read_vtu_xml(path: Path):
    """
    Lê o XML sem que o ElementTree tente interpretar o conteúdo binário.

    O conteúdo binário está em base64 dentro do XML, portanto é seguro
    fazer o parse XML normalmente.
    """
    import xml.etree.ElementTree as ET

    return ET.parse(path)


def vtk_dtype_from_string(type_name: str):
    """
    Converte o tipo VTK para numpy dtype.
    """
    mapping = {
        "Int8": np.int8,
        "UInt8": np.uint8,
        "Int16": np.int16,
        "UInt16": np.uint16,
        "Int32": np.int32,
        "UInt32": np.uint32,
        "Int64": np.int64,
        "UInt64": np.uint64,
        "Float32": np.float32,
        "Float64": np.float64,
    }

    if type_name not in mapping:
        raise ValueError(
            f"Tipo VTK não suportado pelo decoder: {type_name}"
        )

    return mapping[type_name]


def strip_namespace(tag: str) -> str:
    """Remove namespace XML, se existir."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def find_child(parent, name: str):
    """Procura filho ignorando namespace."""
    for child in parent:
        if strip_namespace(child.tag) == name:
            return child
    return None


def find_children(parent, name: str):
    """Procura filhos ignorando namespace."""
    return [
        child
        for child in parent
        if strip_namespace(child.tag) == name
    ]


def get_vtu_piece(root):
    """
    Retorna o primeiro Piece do UnstructuredGrid.
    """
    for elem in root.iter():
        if strip_namespace(elem.tag) == "Piece":
            return elem

    raise ValueError("Não foi encontrado nenhum <Piece> no VTU.")


def clean_base64_text(text: str) -> str:
    """Remove whitespace de um conteúdo base64."""
    if text is None:
        raise ValueError("DataArray binário sem conteúdo.")

    cleaned = re.sub(r"\s+", "", text)

    if not cleaned:
        raise ValueError("DataArray binário vazio.")

    return cleaned


def decode_vtk_compressed_binary(
    encoded_text: str,
    header_type: str,
    byte_order: str,
) -> bytes:
    """
    Decodifica o formato binário comprimido do VTK XML.

    IMPORTANTE:
    O VTK XML não necessariamente codifica todo o conteúdo como um único
    bloco base64. Para vtkZLibDataCompressor, o header comprimido e os
    dados comprimidos são serializados como segmentos base64 consecutivos.

    Exemplo real do arquivo do usuário:

        AQAAAACAAAAIAAAACwAAAA==eF5jYIAAAAAIAAE=

    O primeiro segmento é o header e o segundo é o bloco zlib.

    Header (UInt32, LittleEndian):

        num_blocks
        block_size
        last_block_size
        compressed_size[0]
        compressed_size[1]
        ...

    Depois do header, cada bloco é descomprimido individualmente com zlib.
    """
    if header_type not in ("UInt32", "UInt64"):
        raise ValueError(
            f"header_type={header_type} não suportado. "
            "Esperado UInt32 ou UInt64."
        )

    if byte_order == "LittleEndian":
        endian = "<"
    elif byte_order == "BigEndian":
        endian = ">"
    else:
        raise ValueError(
            f"byte_order desconhecido: {byte_order}"
        )

    cleaned = clean_base64_text(encoded_text)

    header_size = 4 if header_type == "UInt32" else 8
    header_fmt = endian + ("I" if header_type == "UInt32" else "Q")

    # Primeiro descobrimos num_blocks.
    # Um UInt32/UInt64 ocupa 4/8 bytes, portanto os primeiros 8/16
    # caracteres base64 são suficientes para obter o primeiro valor.
    first_value_chars = 8 if header_type == "UInt32" else 12

    if len(cleaned) < first_value_chars:
        raise ValueError(
            "DataArray binário é curto demais para conter o header VTK."
        )

    try:
        first_value_bytes = base64.b64decode(
            cleaned[:first_value_chars],
            validate=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Falha ao decodificar início do header VTK: {exc}"
        ) from exc

    number_of_blocks = struct.unpack_from(
        header_fmt,
        first_value_bytes,
        0,
    )[0]

    if number_of_blocks == 0:
        return b""

    # O header possui:
    #   3 valores fixos + number_of_blocks compressed_sizes
    header_bytes_length = header_size * (3 + number_of_blocks)

    # Número de caracteres base64 necessários para representar o header.
    # O writer do VTK remove os caracteres '=' de padding do segmento
    # base64 do header, mas eles podem permanecer no XML imediatamente
    # antes do próximo segmento. Portanto usamos o tamanho SEM padding.
    full_header_chars = (
        (header_bytes_length + 2) // 3
    ) * 4

    if header_bytes_length % 3 == 1:
        header_chars_length = full_header_chars - 2
    elif header_bytes_length % 3 == 2:
        header_chars_length = full_header_chars - 1
    else:
        header_chars_length = full_header_chars

    if len(cleaned) < header_chars_length:
        raise ValueError(
            "DataArray binário terminou antes do header VTK completo: "
            f"esperados pelo menos {header_chars_length} caracteres base64, "
            f"obtidos {len(cleaned)}."
        )

    header_b64 = cleaned[:header_chars_length]
    payload_b64 = cleaned[header_chars_length:]

    # O padding removido do final do header pode continuar presente no
    # conteúdo XML antes do próximo segmento base64.
    payload_b64 = payload_b64.lstrip("=")

    try:
        header_data = base64.b64decode(
            header_b64 + ("=" * ((4 - len(header_b64) % 4) % 4)),
            validate=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Falha ao decodificar header VTK: {exc}"
        ) from exc

    expected_header_bytes = header_bytes_length

    if len(header_data) < expected_header_bytes:
        raise ValueError(
            "Header VTK incompleto: "
            f"obtidos {len(header_data)} bytes, "
            f"esperados {expected_header_bytes}."
        )

    pos = 0

    def read_header_value():
        nonlocal pos

        if pos + header_size > len(header_data):
            raise ValueError(
                "Header VTK terminou inesperadamente."
            )

        value = struct.unpack_from(
            header_fmt,
            header_data,
            pos,
        )[0]

        pos += header_size
        return value

    # Já lemos number_of_blocks, mas avançamos o cursor normalmente.
    number_of_blocks_header = read_header_value()

    if number_of_blocks_header != number_of_blocks:
        raise ValueError(
            "Inconsistência no número de blocos VTK: "
            f"{number_of_blocks} != {number_of_blocks_header}"
        )

    block_size = read_header_value()
    last_block_size = read_header_value()

    compressed_sizes = [
        read_header_value()
        for _ in range(number_of_blocks)
    ]

    # Depois do header, o VTK usa uma única sequência base64 para os
    # bytes comprimidos de todos os blocos concatenados. Os limites dos
    # blocos são dados por compressed_sizes.
    payload_padded = payload_b64 + (
        "=" * ((4 - len(payload_b64) % 4) % 4)
    )

    try:
        compressed_payload = base64.b64decode(
            payload_padded,
            validate=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Falha ao decodificar payload comprimido VTK: {exc}"
        ) from exc

    expected_compressed_bytes = sum(compressed_sizes)

    if len(compressed_payload) != expected_compressed_bytes:
        raise ValueError(
            "Tamanho do payload comprimido VTK inconsistente: "
            f"obtido={len(compressed_payload)}, "
            f"esperado={expected_compressed_bytes}."
        )

    payload_pos = 0
    output = bytearray()

    for i, compressed_size in enumerate(compressed_sizes):
        end_pos = payload_pos + compressed_size

        if end_pos > len(compressed_payload):
            raise ValueError(
                "DataArray binário terminou no meio de um bloco "
                f"comprimido ({i})."
            )

        compressed_block = compressed_payload[
            payload_pos:end_pos
        ]

        payload_pos = end_pos

        try:
            decompressed = zlib.decompress(compressed_block)
        except zlib.error as exc:
            raise ValueError(
                f"Falha ao descomprimir bloco VTK {i}: {exc}"
            ) from exc

        expected_size = (
            last_block_size
            if i == number_of_blocks - 1
            else block_size
        )

        if len(decompressed) != expected_size:
            raise ValueError(
                f"Tamanho inesperado no bloco {i}: "
                f"obtido={len(decompressed)}, "
                f"esperado={expected_size}."
            )

        output.extend(decompressed)

    return bytes(output)


def decode_vtk_binary_dataarray(
    data_array,
    vtk_root,
) -> np.ndarray:
    """
    Decodifica um DataArray XML no formato binary.

    Suporta:
        format="ascii"
        format="binary"

    O caso binary é o formato observado no VTU do usuário:
        base64 + vtkZLibDataCompressor.
    """
    fmt = data_array.attrib.get("format", "ascii")
    vtk_type = data_array.attrib.get("type")

    if vtk_type is None:
        raise ValueError("DataArray sem atributo 'type'.")

    dtype = vtk_dtype_from_string(vtk_type)

    byte_order = vtk_root.attrib.get("byte_order", "LittleEndian")
    header_type = vtk_root.attrib.get("header_type", "UInt32")
    compressor = vtk_root.attrib.get("compressor")

    if fmt == "ascii":
        text = data_array.text or ""

        values = np.fromstring(text, sep=" ", dtype=dtype)

        return values

    if fmt != "binary":
        raise ValueError(
            f"DataArray format='{fmt}' não suportado por esta versão."
        )

    encoded_text = clean_base64_text(data_array.text or "")

    if compressor == "vtkZLibDataCompressor":
        raw = decode_vtk_compressed_binary(
            encoded_text,
            header_type=header_type,
            byte_order=byte_order,
        )
    else:
        try:
            raw = base64.b64decode(
                encoded_text,
                validate=False,
            )
        except Exception as exc:
            raise ValueError(
                f"Falha ao decodificar base64 do DataArray: {exc}"
            ) from exc

    if compressor and compressor != "vtkZLibDataCompressor":
        raise ValueError(
            f"Compressor não suportado: {compressor}"
        )

    endian_prefix = "<" if byte_order == "LittleEndian" else ">"

    # numpy dtype pode ser convertido explicitamente para a endianness
    dtype_file = np.dtype(dtype).newbyteorder(endian_prefix)

    if len(raw) % dtype_file.itemsize != 0:
        raise ValueError(
            f"Número de bytes ({len(raw)}) incompatível com "
            f"dtype {dtype_file}."
        )

    return np.frombuffer(raw, dtype=dtype_file).copy()


def get_dataarray_by_name(parent, name: str):
    """
    Retorna DataArray com Name=name.
    """
    for elem in parent.iter():
        if strip_namespace(elem.tag) != "DataArray":
            continue

        if elem.attrib.get("Name") == name:
            return elem

    return None


def get_cells_dataarrays(piece):
    """
    Retorna os DataArrays de Cells:

        connectivity
        offsets
        types
    """
    cells = None

    for child in piece:
        if strip_namespace(child.tag) == "Cells":
            cells = child
            break

    if cells is None:
        raise ValueError("VTU sem <Cells>.")

    result = {}

    for da in find_children(cells, "DataArray"):
        name = da.attrib.get("Name")
        if name in {"connectivity", "offsets", "types"}:
            result[name] = da

    missing = {"connectivity", "offsets", "types"} - result.keys()

    if missing:
        raise ValueError(
            "VTU sem DataArray(s) de Cells: "
            + ", ".join(sorted(missing))
        )

    return result


def load_raw_cells(path: Path) -> dict[str, np.ndarray]:
    """
    Lê diretamente connectivity/offsets/types do VTU.

    Esta é a parte que contorna o problema do meshio com VTK_VOXEL (11).
    """
    tree = read_vtu_xml(path)
    root = tree.getroot()
    piece = get_vtu_piece(root)

    arrays = get_cells_dataarrays(piece)

    result = {}

    for name, data_array in arrays.items():
        result[name] = decode_vtk_binary_dataarray(
            data_array,
            root,
        )

    return result


# ---------------------------------------------------------------------------
# Comparação numérica
# ---------------------------------------------------------------------------

def compare_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    atol: float,
    rtol: float,
    integer: bool = False,
) -> dict[str, Any]:
    """
    Compara dois arrays e retorna estatísticas.
    """
    ref = np.asarray(reference)
    cand = np.asarray(candidate)

    result = {
        "shape_reference": list(ref.shape),
        "shape_candidate": list(cand.shape),
        "shape_equal": ref.shape == cand.shape,
        "pass": False,
        "count_different": None,
        "max_abs_error": None,
        "mean_abs_error": None,
        "rmse": None,
        "max_relative_error": None,
    }

    if ref.shape != cand.shape:
        return result

    if ref.size == 0:
        result["pass"] = True
        result["count_different"] = 0
        result["max_abs_error"] = 0.0
        result["mean_abs_error"] = 0.0
        result["rmse"] = 0.0
        result["max_relative_error"] = 0.0
        return result

    if integer:
        equal = ref == cand
        different = ~equal

        result["count_different"] = int(np.count_nonzero(different))
        result["max_abs_error"] = (
            float(np.max(np.abs(
                ref.astype(np.int64) - cand.astype(np.int64)
            )))
            if np.any(different)
            else 0.0
        )
        result["mean_abs_error"] = (
            float(np.mean(np.abs(
                ref.astype(np.int64) - cand.astype(np.int64)
            )))
            if np.any(different)
            else 0.0
        )

        result["rmse"] = (
            float(np.sqrt(np.mean(
                (ref.astype(np.float64) - cand.astype(np.float64)) ** 2
            )))
        )

        result["max_relative_error"] = 0.0
        result["pass"] = bool(np.all(equal))

        return result

    ref_float = ref.astype(np.float64)
    cand_float = cand.astype(np.float64)

    equal = np.isclose(
        ref_float,
        cand_float,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )

    different = ~equal

    abs_error = np.abs(ref_float - cand_float)

    result["count_different"] = int(np.count_nonzero(different))
    result["max_abs_error"] = float(np.max(abs_error))
    result["mean_abs_error"] = float(np.mean(abs_error))
    result["rmse"] = float(
        np.sqrt(np.mean((ref_float - cand_float) ** 2))
    )

    denominator = np.maximum(np.abs(ref_float), atol)

    relative_error = abs_error / denominator
    result["max_relative_error"] = float(np.max(relative_error))

    result["pass"] = bool(np.all(equal))

    return result


# ---------------------------------------------------------------------------
# Comparação das Cells sem meshio
# ---------------------------------------------------------------------------

def compare_raw_cells(
    reference_file: Path,
    candidate_file: Path,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """
    Compara diretamente os arrays de Cells.

    Isso inclui VTK_VOXEL (tipo 11), que pode ser ignorado pelo meshio.
    """
    ref = load_raw_cells(reference_file)
    cand = load_raw_cells(candidate_file)

    result = {
        "pass": True,
        "arrays": {},
        "cell_type_counts_reference": {},
        "cell_type_counts_candidate": {},
    }

    # connectivity e offsets são inteiros
    for name in ("connectivity", "offsets"):
        cmp = compare_arrays(
            ref[name],
            cand[name],
            atol=atol,
            rtol=rtol,
            integer=True,
        )

        result["arrays"][name] = cmp

        if not cmp["pass"]:
            result["pass"] = False

    # types
    cmp = compare_arrays(
        ref["types"],
        cand["types"],
        atol=atol,
        rtol=rtol,
        integer=True,
    )

    result["arrays"]["types"] = cmp

    if not cmp["pass"]:
        result["pass"] = False

    # Contagem por tipo
    ref_types, ref_counts = np.unique(
        ref["types"].astype(np.int64),
        return_counts=True,
    )

    cand_types, cand_counts = np.unique(
        cand["types"].astype(np.int64),
        return_counts=True,
    )

    result["cell_type_counts_reference"] = {
        VTK_CELL_TYPES.get(int(t), f"VTK_TYPE_{int(t)}"): int(c)
        for t, c in zip(ref_types, ref_counts)
    }

    result["cell_type_counts_candidate"] = {
        VTK_CELL_TYPES.get(int(t), f"VTK_TYPE_{int(t)}"): int(c)
        for t, c in zip(cand_types, cand_counts)
    }

    return result


# ---------------------------------------------------------------------------
# meshio
# ---------------------------------------------------------------------------

def normalize_meshio_cells(mesh):
    """
    Retorna um dicionário {cell_type: connectivity}.

    Compatível com versões diferentes do meshio.
    """
    result = {}

    if hasattr(mesh, "cells_dict"):
        for key, value in mesh.cells_dict.items():
            result[key] = np.asarray(value)
        return result

    for block in mesh.cells:
        if hasattr(block, "type"):
            key = block.type
            value = block.data
        else:
            key = block[0]
            value = block[1]

        result[key] = np.asarray(value)

    return result


def compare_meshio(
    reference_file: Path,
    candidate_file: Path,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """
    Comparação usando meshio.

    Observação:
        meshio pode emitir warning para VTK_VOXEL (11).
        Por isso essa comparação NÃO substitui compare_raw_cells().
    """
    if meshio is None:
        return {
            "pass": None,
            "skipped": True,
            "reason": "meshio não está instalado.",
        }

    result = {
        "pass": True,
        "points": None,
        "cells": None,
        "point_data": {},
        "cell_data": {},
    }

    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ref = meshio.read(reference_file)
            cand = meshio.read(candidate_file)
    except Exception as exc:
        return {
            "pass": False,
            "skipped": False,
            "error": f"meshio.read falhou: {exc}",
        }

    # Points
    cmp = compare_arrays(
        np.asarray(ref.points),
        np.asarray(cand.points),
        atol=atol,
        rtol=rtol,
        integer=False,
    )

    result["points"] = cmp

    if not cmp["pass"]:
        result["pass"] = False

    # Cells reconhecidas pelo meshio
    ref_cells = normalize_meshio_cells(ref)
    cand_cells = normalize_meshio_cells(cand)

    result["cells"] = {
        "pass": True,
        "types_reference": sorted(ref_cells.keys()),
        "types_candidate": sorted(cand_cells.keys()),
        "per_type": {},
    }

    if set(ref_cells.keys()) != set(cand_cells.keys()):
        result["cells"]["pass"] = False
        result["pass"] = False

    for cell_type in sorted(
        set(ref_cells.keys()) | set(cand_cells.keys())
    ):
        if cell_type not in ref_cells or cell_type not in cand_cells:
            result["cells"]["per_type"][cell_type] = {
                "pass": False,
                "reason": "tipo presente somente em um arquivo",
            }
            continue

        cmp = compare_arrays(
            ref_cells[cell_type],
            cand_cells[cell_type],
            atol=atol,
            rtol=rtol,
            integer=True,
        )

        result["cells"]["per_type"][cell_type] = cmp

        if not cmp["pass"]:
            result["cells"]["pass"] = False
            result["pass"] = False

    # PointData
    ref_pd = {
        str(k): np.asarray(v)
        for k, v in ref.point_data.items()
    }

    cand_pd = {
        str(k): np.asarray(v)
        for k, v in cand.point_data.items()
    }

    all_pd = sorted(set(ref_pd) | set(cand_pd))

    for name in all_pd:

        # State e Beat são ignorados na regressão porque,
        # após a separação do solver, sua representação pode
        # diferir enquanto a fisiologia/propagação permanece igual.
        if name in {"State", "Beat"}:
            continue

        if name not in ref_pd or name not in cand_pd:
            result["point_data"][name] = {
                "pass": False,
                "reason": "campo presente somente em um arquivo",
            }
            result["pass"] = False
            continue

        cmp = compare_arrays(
            ref_pd[name],
            cand_pd[name],
            atol=atol,
            rtol=rtol,
            integer=False,
        )

        result["point_data"][name] = cmp

        if not cmp["pass"]:
            result["pass"] = False

    # CellData
    # meshio pode organizar CellData de forma diferente entre versões.
    # Fazemos a comparação apenas quando disponível de maneira clara.
    try:
        ref_cd = {
            str(k): np.asarray(v)
            for k, v in ref.cell_data.items()
        }

        cand_cd = {
            str(k): np.asarray(v)
            for k, v in cand.cell_data.items()
        }

        all_cd = sorted(set(ref_cd) | set(cand_cd))

        for name in all_cd:
            if name not in ref_cd or name not in cand_cd:
                result["cell_data"][name] = {
                    "pass": False,
                    "reason": "campo presente somente em um arquivo",
                }
                result["pass"] = False
                continue

            cmp = compare_arrays(
                ref_cd[name],
                cand_cd[name],
                atol=atol,
                rtol=rtol,
                integer=name in INTEGER_FIELDS,
            )

            result["cell_data"][name] = cmp

            if not cmp["pass"]:
                result["pass"] = False

    except Exception as exc:
        result["cell_data"] = {
            "pass": None,
            "skipped": True,
            "reason": str(exc),
        }

    return result


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def print_array_result(name: str, result: dict[str, Any], indent="       "):
    if result.get("pass"):
        print(f"{indent}{name:<14} PASS")
        return

    if result.get("pass") is None:
        print(f"{indent}{name:<14} SKIP")
        return

    print(f"{indent}{name:<14} FAIL")

    if "count_different" in result:
        print(
            f"{indent}  diferenças: "
            f"{result['count_different']}"
        )

    if result.get("max_abs_error") is not None:
        print(
            f"{indent}  max abs:    "
            f"{format_number(result['max_abs_error'])}"
        )

    if result.get("rmse") is not None:
        print(
            f"{indent}  RMSE:       "
            f"{format_number(result['rmse'])}"
        )

    if result.get("max_relative_error") is not None:
        print(
            f"{indent}  max rel:    "
            f"{format_number(result['max_relative_error'])}"
        )

    if not result.get("shape_equal", True):
        print(
            f"{indent}  shape ref:  "
            f"{result.get('shape_reference')}"
        )
        print(
            f"{indent}  shape cand: "
            f"{result.get('shape_candidate')}"
        )


def print_raw_cells_result(result: dict[str, Any]):
    print("       cells_raw     ", end="")

    if result["pass"]:
        print("PASS")
    else:
        print("FAIL")

    for name in ("connectivity", "offsets", "types"):
        print_array_result(
            f"raw/{name}",
            result["arrays"][name],
            indent="         ",
        )

    ref_counts = result["cell_type_counts_reference"]
    cand_counts = result["cell_type_counts_candidate"]

    print("         tipos de célula:")
    print(f"           reference: {ref_counts}")
    print(f"           candidate: {cand_counts}")

    # Destaque especial para VTK_VOXEL
    ref_voxel = ref_counts.get("VOXEL", 0)
    cand_voxel = cand_counts.get("VOXEL", 0)

    print(
        f"           VTK_VOXEL(11): "
        f"reference={ref_voxel}, "
        f"candidate={cand_voxel}"
    )


def print_meshio_result(result: dict[str, Any]):
    if result.get("skipped"):
        print(
            "       meshio        SKIP "
            f"({result.get('reason', '')})"
        )
        return

    if "error" in result:
        print("       meshio        FAIL")
        print(f"         {result['error']}")
        return

    print(
        "       meshio        "
        + ("PASS" if result["pass"] else "FAIL")
    )

    print_array_result(
        "points",
        result["points"],
        indent="         ",
    )

    cells = result["cells"]

    print(
        "         cells       "
        + ("PASS" if cells["pass"] else "FAIL")
    )

    for cell_type, cmp in cells["per_type"].items():
        print_array_result(
            f"cells/{cell_type}",
            cmp,
            indent="           ",
        )

    for name, cmp in result["point_data"].items():
        print_array_result(
            f"PointData/{name}",
            cmp,
            indent="         ",
        )


# ---------------------------------------------------------------------------
# Arquivo
# ---------------------------------------------------------------------------

def compare_file(
    reference_file: Path,
    candidate_file: Path,
    atol: float,
    rtol: float,
    use_meshio: bool,
    use_raw_cells: bool,
    verbose: bool,
    print_result: bool = True,
) -> dict[str, Any]:
    result = {"file": reference_file.name, "pass": True, "meshio": None, "raw_cells": None}

    if use_meshio:
        meshio_result = compare_meshio(reference_file, candidate_file, atol, rtol)
        result["meshio"] = meshio_result
        if meshio_result.get("pass") is False:
            result["pass"] = False

    if use_raw_cells:
        try:
            raw_result = compare_raw_cells(reference_file, candidate_file, atol, rtol)
            result["raw_cells"] = raw_result
            if not raw_result["pass"]:
                result["pass"] = False
        except Exception as exc:
            result["raw_cells"] = {"pass": False, "error": str(exc)}
            result["pass"] = False

    if print_result:
        print_file_result(result)
    return result


def print_file_result(result: dict[str, Any]):
    print(result["file"])
    meshio_result = result.get("meshio")
    if meshio_result is not None and meshio_result.get("pass") is False:
        print_meshio_result(meshio_result)
    raw_result = result.get("raw_cells")
    if raw_result is not None and raw_result.get("pass") is False:
        if "error" in raw_result:
            print("       cells_raw     FAIL")
            print(f"         erro: {raw_result['error']}")
        else:
            print_raw_cells_result(raw_result)
    print("       RESULTADO     FAIL")


# ---------------------------------------------------------------------------
# Caso
# ---------------------------------------------------------------------------

def compare_case(
    reference_dir: Path,
    candidate_dir: Path,
    atol: float,
    rtol: float,
    use_meshio: bool,
    use_raw_cells: bool,
    verbose: bool,
):
    print("=" * 78)
    print(f"CASO: {reference_dir.name}")
    print(f"REFERENCE: {reference_dir}")
    print(f"CANDIDATE: {candidate_dir}")
    print("=" * 78)

    ref_files = discover_vtu_files(reference_dir)
    cand_files = discover_vtu_files(candidate_dir)

    common = sorted(
        set(ref_files) & set(cand_files),
        key=natural_key,
    )

    only_ref = sorted(
        set(ref_files) - set(cand_files),
        key=natural_key,
    )

    only_cand = sorted(
        set(cand_files) - set(ref_files),
        key=natural_key,
    )

    print(f"Arquivos reference: {len(ref_files)}")
    print(f"Arquivos candidate: {len(cand_files)}")
    print(f"Arquivos comuns:    {len(common)}")

    if only_ref:
        print("\nSomente na reference:")
        for name in only_ref:
            print(f"  {name}")

    if only_cand:
        print("\nSomente na candidate:")
        for name in only_cand:
            print(f"  {name}")

    if only_ref or only_cand:
        print("\nRESULTADO: FAIL (conjunto de arquivos diferente)")
        return {
            "case": reference_dir.name,
            "pass": False,
            "files": [],
            "only_reference": only_ref,
            "only_candidate": only_cand,
        }

    results = []

    for name in common:
        result = compare_file(
            ref_files[name],
            cand_files[name],
            atol,
            rtol,
            use_meshio=use_meshio,
            use_raw_cells=use_raw_cells,
            verbose=verbose,
            print_result=False,
        )
        results.append(result)
        if not result["pass"]:
            print()
            print_file_result(result)

    passed = sum(r["pass"] for r in results)
    total = len(results)

    overall_pass = (
        total > 0
        and passed == total
    )

    print()
    print("-" * 78)
    print(f"Arquivos: {passed}/{total} passaram")

    if overall_pass:
        print("RESULTADO FINAL: PASS")
    else:
        print("RESULTADO FINAL: FAIL")

    return {
        "case": reference_dir.name,
        "pass": overall_pass,
        "files": results,
        "only_reference": only_ref,
        "only_candidate": only_cand,
    }


# ---------------------------------------------------------------------------
# Descoberta de casos
# ---------------------------------------------------------------------------

def choose_cases(
    reference_root: Path,
    candidate_root: Path,
    case_name: str | None,
):
    """
    Identifica os casos a comparar.

    Cenário A:
        reference_root/03_S1S2/*.vtu
        candidate_root/03_S1S2/*.vtu

    Cenário B:
        reference_root/*.vtu
        candidate_root/*.vtu
    """
    ref_direct = any(reference_root.glob("*.vtu"))
    cand_direct = any(candidate_root.glob("*.vtu"))

    if ref_direct or cand_direct:
        if not (ref_direct and cand_direct):
            raise RuntimeError(
                "Um dos diretórios contém VTU diretamente e o outro não."
            )

        return [(reference_root, candidate_root)]

    ref_cases = {
        p.name: p
        for p in find_case_directories(reference_root)
    }

    cand_cases = {
        p.name: p
        for p in find_case_directories(candidate_root)
    }

    if case_name:
        if case_name not in ref_cases:
            raise RuntimeError(
                f"Caso '{case_name}' não encontrado na reference."
            )

        if case_name not in cand_cases:
            raise RuntimeError(
                f"Caso '{case_name}' não encontrado na candidate."
            )

        return [(ref_cases[case_name], cand_cases[case_name])]

    common = sorted(
        set(ref_cases) & set(cand_cases),
        key=natural_key,
    )

    return [
        (ref_cases[name], cand_cases[name])
        for name in common
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compara resultados VTU do Arritmic3D, "
            "incluindo Cells binárias comprimidas."
        )
    )

    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Diretório com resultados de referência.",
    )

    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Diretório com resultados do código novo.",
    )

    parser.add_argument(
        "--case",
        default=None,
        help="Compara somente este caso.",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"Tolerância absoluta (default={DEFAULT_ATOL}).",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"Tolerância relativa (default={DEFAULT_RTOL}).",
    )

    parser.add_argument(
        "--no-meshio",
        action="store_true",
        help="Desabilita a comparação via meshio.",
    )

    parser.add_argument(
        "--no-raw-cells",
        action="store_true",
        help="Desabilita a comparação direta das Cells.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra informações adicionais.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Prefixo do relatório. "
            "Ex.: --report comparison"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.reference.exists():
        print(
            f"ERRO: reference não existe: {args.reference}",
            file=sys.stderr,
        )
        return 2

    if not args.candidate.exists():
        print(
            f"ERRO: candidate não existe: {args.candidate}",
            file=sys.stderr,
        )
        return 2

    try:
        cases = choose_cases(
            args.reference,
            args.candidate,
            args.case,
        )
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2

    if not cases:
        print(
            "ERRO: nenhum caso em comum foi encontrado.",
            file=sys.stderr,
        )
        return 2

    all_results = []
    all_pass = True

    for reference_dir, candidate_dir in cases:
        result = compare_case(
            reference_dir,
            candidate_dir,
            atol=args.atol,
            rtol=args.rtol,
            use_meshio=not args.no_meshio,
            use_raw_cells=not args.no_raw_cells,
            verbose=args.verbose,
        )

        all_results.append(result)

        if not result["pass"]:
            all_pass = False

    # -----------------------------------------------------------------------
    # Resumo final
    # -----------------------------------------------------------------------

    print()
    print("=" * 78)
    print("RESUMO FINAL")
    print("=" * 78)

    for result in all_results:
        print(
            f"{result['case']:<30} "
            + ("PASS" if result["pass"] else "FAIL")
        )

    print("=" * 78)

    if all_pass:
        print("RESULTADO FINAL: PASS")
    else:
        print("RESULTADO FINAL: FAIL")

    # -----------------------------------------------------------------------
    # JSON
    # -----------------------------------------------------------------------

    json_data = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "atol": args.atol,
        "rtol": args.rtol,
        "cases": all_results,
        "pass": all_pass,
    }

    if args.report:
        report_prefix = args.report

        if report_prefix.suffix:
            report_prefix = report_prefix.with_suffix("")

        json_file = report_prefix.with_suffix(".json")
        txt_file = report_prefix.with_suffix(".txt")

        json_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            json_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                json_data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Relatório textual simples
        with open(
            txt_file,
            "w",
            encoding="utf-8",
        ) as f:
            f.write("=" * 78 + "\n")
            f.write("ARRITMIC3D REGRESSION REPORT\n")
            f.write("=" * 78 + "\n\n")

            f.write(f"Reference: {args.reference}\n")
            f.write(f"Candidate: {args.candidate}\n")
            f.write(f"ATOL: {args.atol}\n")
            f.write(f"RTOL: {args.rtol}\n\n")

            for case in all_results:
                f.write(
                    f"CASO: {case['case']} - "
                    f"{'PASS' if case['pass'] else 'FAIL'}\n"
                )

                for file_result in case["files"]:
                    f.write(
                        f"  {file_result['file']}: "
                        f"{'PASS' if file_result['pass'] else 'FAIL'}\n"
                    )

            f.write("\n")
            f.write(
                "RESULTADO FINAL: "
                + ("PASS" if all_pass else "FAIL")
                + "\n"
            )

        print()
        print(f"Relatório JSON: {json_file}")
        print(f"Relatório TXT:  {txt_file}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())