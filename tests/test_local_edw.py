from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import duckdb

from app.data_access.local_database import build_local_database, database_is_current
from app.data_access.repository import LocalCsvRepository


class LocalEdwTest(unittest.TestCase):
    def _write_csv(self, root: Path, name: str, columns: list[str], rows: list[list[object]]) -> None:
        with (root / name).open("w", encoding="cp1251", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_builder_and_repository_use_schema_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "database" / "edw.db"
            # Simulate the previous manifest + per-schema SQLite layout.
            (db_path.parent / "schemas").mkdir(parents=True)
            (db_path.parent / "manifests").mkdir()
            db_path.write_bytes(b"legacy SQLite placeholder")
            (db_path.parent / "schemas" / "old.db").write_bytes(b"old schema")
            (db_path.parent / "manifests" / "old.db").write_bytes(b"old manifest")
            self._write_csv(
                root,
                "dp_rid_ods_corp_oisproduction_db5555.arx.csv",
                ["sk_1", "gg_1", "dd_1", "ms_1", "pl_1", "xr_1", "sm_1", "tr_1", "qn_1", "qv_1", "dn_1", "dv_1", "qj_1"],
                [["100", "2025", "1", "MS0001", "PL0001", "XR0011", "SM0001", "30", "10", "5", "300", "150", "0"]],
            )
            self._write_csv(
                root,
                "dp_rid_ods_corp_oisproduction_db5555.class.csv",
                ["cd_1", "ne_1", "ns_1"],
                [
                    ["MS0001", "Тестовое", "Тест"],
                    ["PL0001", "Ю1", "Ю1"],
                    ["XR0011", "Добывающая", "НЕФ"],
                    ["SM0001", "Рабочая", "РАБ."],
                ],
            )
            self._write_csv(
                root,
                "dp_rid_ods_corp_oisproduction_db5555.spskv.csv",
                ["sk_1", "xx_1", "yy_1"],
                [["100", "10", "20"]],
            )
            self._write_csv(
                root,
                "dp_rid_ods_corp_geobd_dwh.geooistranslation.csv",
                ["uwi", "sk_1", "smart_name"],
                [["UWI-100", "100", "101"]],
            )
            self._write_csv(
                root,
                "dp_rid_ods_corp_geobd_super_man.plastcross.csv",
                ["uwi", "orig_layers_name", "object_name", "node_x", "node_y"],
                [["UWI-100", "well_T3", "Ю1", "15", "25"]],
            )
            self._write_csv(
                root,
                "dp_rid_ods_corp_shtr_wellop.well_layer_op.csv",
                ["well_id", "layer_id", "calc_date", "oil_rate", "liq_rate", "water_cut", "layer_shut_pressure", "layer_flow_pressure"],
                [["100", "PL0001", "2025-01-15 00:00:00", "9", "14", "35", "100", "90"]],
            )
            self._write_csv(
                root,
                "dp_rid_ods_corp_shtr_wellop.well_layer_op_inj.csv",
                ["well_id", "layer_id", "calc_date", "intake", "layer_shut_pressure", "layer_flow_pressure"],
                [["100", "PL0001", "2025-01-15 00:00:00", "0", "0", "0"]],
            )
            self._write_csv(
                root,
                "dp_rid_ods_corp_geobd_super_man.grp_info.csv",
                ["uwi", "date_op", "layer", "kpr", "ppl", "xf", "wf", "skin", "top", "bot", "comments", "num_oper"],
                [["UWI-100", "2024-01-01", "Ю1", "50", "200", "100", "2", "-1", "1000", "1010", "test", "1"]],
            )
            self._write_csv(
                root,
                "dp_rid_ods_corp_ngtsmart_rdfsmart.pvt.csv",
                ["oil_field", "layer", "initialpressure", "permeability"],
                [["Тестовое", "Ю1", "225", "50"]],
            )
            grid_columns = [
                "gridkey", "name", "gridtype", "dt", "field", "object", "data",
                "countx", "county", "blankvalue", "offsetx", "offsety", "stepx", "stepy",
                "x_max", "x_min", "y_max", "y_min", "coordsys",
            ]
            self._write_csv(
                root,
                "dp_rid_ods_corp_eragrad_api.gridlist.csv",
                grid_columns,
                [[
                    "grid-1", "ННТ", "Карта начальной нефтенасыщенной толщины", "2025-01-01",
                    "Тестовое", "Ю1", json.dumps([[1, 2], [3, 4]]), "2", "2", "-9999",
                    "0", "10", "1", "1", "1", "0", "10", "9", "",
                ]],
            )
            # This query result must never become a database table.
            self._write_csv(
                root,
                "field_objects_keys.csv",
                ["field", "ms_1", "objects", "pl_1"],
                [["Ошибочное", "BAD", "BAD", "BAD"]],
            )

            build_local_database(root, db_path, progress=lambda _message: None)
            self.assertTrue(database_is_current(root, db_path))
            self.assertEqual(list(db_path.parent.glob("*.db")), [db_path])
            self.assertFalse((db_path.parent / "manifests").exists())
            self.assertFalse((db_path.parent / "schemas").exists())
            with closing(duckdb.connect(str(db_path), read_only=True)) as database:
                schemas = {
                    row[0]
                    for row in database.execute(
                        """SELECT schema_name FROM information_schema.schemata
                           WHERE schema_name LIKE 'dp_rid_ods_corp_%'"""
                    ).fetchall()
                }
                self.assertIn("dp_rid_ods_corp_eragrad_api", schemas)
                self.assertEqual(
                    database.execute(
                        """SELECT COUNT(*)
                           FROM edw.dp_rid_ods_corp_eragrad_api.gridlist"""
                    ).fetchone()[0],
                    1,
                )

            # A closed database is replaced in place; no second version remains.
            build_local_database(
                root, db_path, force=True, progress=lambda _message: None
            )
            self.assertTrue(database_is_current(root, db_path))
            self.assertEqual(list(db_path.parent.glob("*.db")), [db_path])

            repository = LocalCsvRepository(root, db_path)
            keys = repository.list_field_objects()
            self.assertEqual(keys[["field", "objects"]].to_dict("records"), [{"field": "Тестовое", "objects": "Ю1"}])
            history = repository.get_history("Тестовое", "Ю1")
            self.assertEqual(history.iloc[0]["well_number"], "101")
            self.assertEqual(history.iloc[0]["T3_x_geo"], 15.0)
            self.assertEqual(len(repository.get_fracs("Тестовое", "Ю1")), 1)
            self.assertTrue(repository.get_pvt("Тестовое", "Ю1")["found"])
            grids = repository.list_grids("Тестовое", "Ю1")
            self.assertEqual(len(grids), 1)
            self.assertEqual(repository.get_grid(grids[0].locator)["data"], "[[1, 2], [3, 4]]")


if __name__ == "__main__":
    unittest.main()
