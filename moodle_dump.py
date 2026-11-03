"""
Parser Dump SQL Moodle (MariaDB/MySQL) -> tabel pandas

API WS Moodle sekolah tidak tersedia, sehingga data diambil dari file dump
(mysqldump/mariadb-dump). Modul ini membaca CREATE TABLE untuk nama kolom dan
INSERT INTO untuk isinya, tanpa perlu memasang server MySQL.

Contoh:
    python moodle_dump.py ../exp_kokar.sql data_kokar/tables
"""

import os
import re
import sys

import pandas as pd

# Tabel Moodle yang dipakai untuk ekstraksi 24 fitur EWS
TABLES = [
    "mdl_user", "mdl_course", "mdl_context", "mdl_role_assignments", "mdl_user_enrolments", "mdl_enrol",
    "mdl_cohort", "mdl_cohort_members", "mdl_logstore_standard_log", "mdl_course_modules", "mdl_modules",
    "mdl_quiz", "mdl_quiz_attempts", "mdl_quiz_grades", "mdl_assign", "mdl_assign_submission",
    "mdl_assign_grades", "mdl_lesson", "mdl_grade_items", "mdl_grade_grades",
]

USER_COLS = ["id", "username", "firstname", "lastname", "deleted", "suspended"]

_ESC = {"n": "\n", "r": "\r", "t": "\t", "0": "\0", "Z": "\x1a"}


def _parse_values(s, i):
    """Membaca rangkaian tuple '(..),(..);' mulai dari indeks i."""
    rows, n = [], len(s)
    while i < n:
        c = s[i]
        if c == "(":
            i += 1
            row = []
            while True:
                if s[i] == "'":
                    i += 1
                    buf = []
                    while True:
                        c = s[i]
                        if c == "\\":
                            buf.append(_ESC.get(s[i + 1], s[i + 1]))
                            i += 2
                        elif c == "'":
                            if s[i + 1] == "'":
                                buf.append("'")
                                i += 2
                            else:
                                i += 1
                                break
                        else:
                            j = i
                            while s[j] not in "\\'":
                                j += 1
                            buf.append(s[i:j])
                            i = j
                    row.append("".join(buf))
                else:
                    j = i
                    while s[j] not in ",)":
                        j += 1
                    tok = s[i:j].strip()
                    i = j
                    if tok == "NULL":
                        row.append(None)
                    else:
                        try:
                            row.append(int(tok))
                        except ValueError:
                            try:
                                row.append(float(tok))
                            except ValueError:
                                row.append(tok)
                if s[i] == ",":
                    i += 1
                    continue
                i += 1  # ')'
                break
            rows.append(row)
        elif c == ";":
            break
        else:
            i += 1
    return rows


def load_dump(path, tables=None):
    """Mengembalikan dict {nama_tabel: DataFrame} untuk tabel yang diminta (semua jika None)."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    cols = {m.group(1): re.findall(r"^\s+`(\w+)`", m.group(2), re.M)
            for m in re.finditer(r"CREATE TABLE `(\w+)` \((.*?)\n\)", txt, re.S)}
    data = {}
    for m in re.finditer(r"INSERT INTO `(\w+)` VALUES", txt):
        t = m.group(1)
        if tables and t not in tables:
            continue
        data.setdefault(t, []).extend(_parse_values(txt, m.end()))
    wanted = tables or list(cols)
    return {t: pd.DataFrame(data.get(t, []), columns=cols[t]) for t in wanted if t in cols}


def load_tables(dump_path, cache_dir=None, tables=TABLES):
    """Membaca tabel dari cache pickle bila ada, jika tidak mem-parse dump lalu menyimpan cache."""
    if cache_dir and all(os.path.exists(os.path.join(cache_dir, t + ".pkl")) for t in tables):
        return {t: pd.read_pickle(os.path.join(cache_dir, t + ".pkl")) for t in tables}
    out = load_dump(dump_path, tables)
    if "mdl_user" in out:  # buang password, email, telepon, alamat, dll. yang tidak dipakai model
        out["mdl_user"] = out["mdl_user"][USER_COLS]
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for t, df in out.items():
            df.to_pickle(os.path.join(cache_dir, t + ".pkl"))
    return out


if __name__ == "__main__":
    tabs = load_tables(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    for t, df in tabs.items():
        print(f"{t:35s} {len(df):>8}")
