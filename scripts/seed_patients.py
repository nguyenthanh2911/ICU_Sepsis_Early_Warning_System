from __future__ import annotations

import os
import random
from typing import List, Tuple

import psycopg2
from dotenv import load_dotenv


PATIENTS: List[Tuple[str, str]] = [
    ("Nguyễn Văn An",    "M"),
    ("Trần Thị Bình",    "F"),
    ("Lê Văn Cường",     "M"),
    ("Phạm Thị Dung",    "F"),
    ("Hoàng Văn Đức",    "M"),
    ("Vũ Thị Hạnh",      "F"),
    ("Đặng Văn Hùng",    "M"),
    ("Bùi Thị Lan",      "F"),
    ("Ngô Văn Minh",     "M"),
    ("Đỗ Thị Nga",       "F"),
    ("Hồ Văn Phúc",      "M"),
    ("Dương Thị Quỳnh",  "F"),
    ("Lý Văn Sơn",       "M"),
    ("Mai Thị Trang",    "F"),
    ("Đinh Văn Tuấn",    "M"),
    ("Chu Thị Vân",      "F"),
    ("Tạ Văn Xuyên",     "M"),
    ("Ninh Thị Yến",     "F"),
    ("Cao Văn Khoa",     "M"),
    ("Hà Thị Liên",      "F"),
    ("Tôn Văn Nam",      "M"),
    ("Lâm Thị Oanh",     "F"),
    ("Kiều Văn Quang",   "M"),
    ("Phan Thị Thu",     "F"),
]


def _conn_params_from_env() -> dict:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "sepsis_user")
    password = os.getenv("POSTGRES_PASSWORD", "sepsis_pass")
    db = os.getenv("POSTGRES_DB", "sepsis_db")
    return {"host": host, "port": port, "user": user, "password": password, "dbname": db}


def main() -> None:
    load_dotenv()

    params = _conn_params_from_env()

    sampled = random.sample(PATIENTS, 20)

    patients: List[Tuple[str, str, int, str, str]] = []
    for i, pid in enumerate(f"P{i+1:03d}" for i in range(20)):
        name, gender = sampled[i]
        age = random.randint(45, 80)
        ward = random.choice(["ICU-1", "ICU-2"])
        patients.append((pid, name, age, gender, ward))

    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO patients (patient_id, name, age, gender, ward)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (patient_id) DO NOTHING
                """,
                patients,
            )
        conn.commit()

    print("Seeded 20 patients (P001..P020)")


if __name__ == "__main__":
    main()
