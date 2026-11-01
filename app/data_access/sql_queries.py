"""SQL shared by the one-file DuckDB imitation and the KSPD Greenplum EDW.

Both backends expose the same ``schema.table`` names. Keep these statements
limited to SQL constructs supported by DuckDB and Greenplum.
"""

FIELD_OBJECTS_SQL = """
SELECT DISTINCT
    ms.ne_1 AS field,
    mer.ms_1,
    pl.ne_1 AS objects,
    mer.pl_1
FROM dp_rid_ods_corp_oisproduction_db5555.arx AS mer
JOIN dp_rid_ods_corp_oisproduction_db5555.class AS ms
  ON mer.ms_1 = ms.cd_1
JOIN dp_rid_ods_corp_oisproduction_db5555.class AS pl
  ON mer.pl_1 = pl.cd_1
WHERE TRIM(COALESCE(ms.ne_1, '')) <> ''
  AND TRIM(COALESCE(pl.ne_1, '')) <> ''
ORDER BY field, objects
"""


SELECTED_KEY_SQL = """
SELECT DISTINCT
    mer.ms_1,
    mer.pl_1
FROM dp_rid_ods_corp_oisproduction_db5555.arx AS mer
JOIN dp_rid_ods_corp_oisproduction_db5555.class AS ms
  ON mer.ms_1 = ms.cd_1
JOIN dp_rid_ods_corp_oisproduction_db5555.class AS pl
  ON mer.pl_1 = pl.cd_1
WHERE LOWER(TRIM(ms.ne_1)) = LOWER(TRIM(:field))
  AND LOWER(TRIM(pl.ne_1)) = LOWER(TRIM(:object_name))
LIMIT 1
"""


HISTORY_SQL = """
SELECT DISTINCT
    CAST(mer.sk_1 AS TEXT) AS _well_id,
    ms.ne_1 AS field,
    COALESCE(NULLIF(CAST(g.smart_name AS TEXT), ''), CAST(mer.sk_1 AS TEXT)) AS well_number,
    CAST(mer.gg_1 AS INTEGER) AS _year,
    CAST(mer.dd_1 AS INTEGER) AS _month,
    xr.ns_1 AS work_marker,
    sm.ns_1 AS well_status,
    pl.ne_1 AS object,
    pl.ne_1 AS objects,
    CAST(COALESCE(mer.tr_1, '0') AS REAL) AS time_work,
    CASE WHEN mer.xr_1 = 'XR0011' THEN CAST(COALESCE(mer.tr_1, '0') AS REAL) ELSE 0 END AS time_work_prod,
    CASE WHEN mer.xr_1 = 'XR0020' THEN CAST(COALESCE(mer.tr_1, '0') AS REAL) ELSE 0 END AS time_work_inj,
    CAST(COALESCE(mer.qn_1, '0') AS REAL) AS "Qo_rate",
    CAST(COALESCE(wlo.oil_rate, '0') AS REAL) AS "Qo_rate_TR",
    CAST(COALESCE(mer.qn_1, '0') AS REAL) + CAST(COALESCE(mer.qv_1, '0') AS REAL) AS "Ql_rate",
    CAST(COALESCE(wlo.liq_rate, '0') AS REAL) AS "Ql_rate_TR",
    CASE WHEN mer.xr_1 = 'XR0011' THEN CAST(COALESCE(mer.dn_1, '0') AS REAL) ELSE 0 END AS "Qo",
    CAST(COALESCE(mer.dn_1, '0') AS REAL) + CAST(COALESCE(mer.dv_1, '0') AS REAL) AS "Ql",
    CASE
      WHEN CAST(COALESCE(mer.qn_1, '0') AS REAL) + CAST(COALESCE(mer.qv_1, '0') AS REAL) > 0
      THEN CAST(COALESCE(mer.qv_1, '0') AS REAL) * 100.0
           / (CAST(COALESCE(mer.qn_1, '0') AS REAL) + CAST(COALESCE(mer.qv_1, '0') AS REAL))
      ELSE 0
    END AS water_cut,
    CAST(COALESCE(wlo.water_cut, '0') AS REAL) AS "water_cut_TR",
    CAST(COALESCE(mer.qj_1, '0') AS REAL) AS "Winj_rate",
    CAST(COALESCE(wloi.intake, '0') AS REAL) AS "Winj_rate_TR",
    CASE WHEN mer.xr_1 = 'XR0020' THEN CAST(COALESCE(mer.dn_1, '0') AS REAL) ELSE 0 END AS "Winj",
    CASE
      WHEN mer.xr_1 = 'XR0011' THEN CAST(COALESCE(wlo.layer_shut_pressure, '0') AS REAL)
      WHEN mer.xr_1 = 'XR0020' THEN CAST(COALESCE(wloi.layer_shut_pressure, '0') AS REAL)
      ELSE 0
    END AS "P_well",
    CASE
      WHEN mer.xr_1 = 'XR0011' THEN CAST(COALESCE(wlo.layer_flow_pressure, '0') AS REAL)
      WHEN mer.xr_1 = 'XR0020' THEN CAST(COALESCE(wloi.layer_flow_pressure, '0') AS REAL)
      ELSE 0
    END AS "P_reservoir",
    CAST(COALESCE(info_wells.xx_1, '0') AS REAL) AS "T1_x_geo",
    CAST(COALESCE(info_wells.yy_1, '0') AS REAL) AS "T1_y_geo",
    CAST(COALESCE(pc.node_x, info_wells.xx_1, '0') AS REAL) AS "T3_x_geo",
    CAST(COALESCE(pc.node_y, info_wells.yy_1, '0') AS REAL) AS "T3_y_geo",
    CAST(COALESCE(info_wells.xx_1, '0') AS REAL) AS "T_head_x_geo",
    CAST(COALESCE(info_wells.yy_1, '0') AS REAL) AS "T_head_y_geo"
FROM dp_rid_ods_corp_oisproduction_db5555.arx AS mer
JOIN dp_rid_ods_corp_oisproduction_db5555.class AS ms
  ON mer.ms_1 = ms.cd_1
JOIN dp_rid_ods_corp_oisproduction_db5555.class AS pl
  ON mer.pl_1 = pl.cd_1
JOIN dp_rid_ods_corp_oisproduction_db5555.class AS xr
  ON mer.xr_1 = xr.cd_1
JOIN dp_rid_ods_corp_oisproduction_db5555.class AS sm
  ON mer.sm_1 = sm.cd_1
LEFT JOIN (
    SELECT sk_1, MAX(smart_name) AS smart_name, MAX(uwi) AS uwi
    FROM dp_rid_ods_corp_geobd_dwh.geooistranslation
    GROUP BY sk_1
) AS g ON mer.sk_1 = g.sk_1
LEFT JOIN (
    SELECT sk_1, MAX(xx_1) AS xx_1, MAX(yy_1) AS yy_1
    FROM dp_rid_ods_corp_oisproduction_db5555.spskv
    GROUP BY sk_1
) AS info_wells ON mer.sk_1 = info_wells.sk_1
LEFT JOIN (
    SELECT
        well_id,
        layer_id,
        CAST(SUBSTR(CAST(calc_date AS TEXT), 1, 4) AS INTEGER) AS calc_year,
        CAST(SUBSTR(CAST(calc_date AS TEXT), 6, 2) AS INTEGER) AS calc_month,
        MAX(oil_rate) AS oil_rate,
        MAX(liq_rate) AS liq_rate,
        MAX(water_cut) AS water_cut,
        MAX(layer_shut_pressure) AS layer_shut_pressure,
        MAX(layer_flow_pressure) AS layer_flow_pressure
    FROM dp_rid_ods_corp_shtr_wellop.well_layer_op
    WHERE layer_id = :pl_code
    GROUP BY well_id, layer_id, calc_year, calc_month
) AS wlo
  ON mer.sk_1 = wlo.well_id
 AND mer.pl_1 = wlo.layer_id
 AND CAST(mer.gg_1 AS INTEGER) = wlo.calc_year
 AND CAST(mer.dd_1 AS INTEGER) = wlo.calc_month
LEFT JOIN (
    SELECT
        well_id,
        layer_id,
        CAST(SUBSTR(CAST(calc_date AS TEXT), 1, 4) AS INTEGER) AS calc_year,
        CAST(SUBSTR(CAST(calc_date AS TEXT), 6, 2) AS INTEGER) AS calc_month,
        MAX(intake) AS intake,
        MAX(layer_shut_pressure) AS layer_shut_pressure,
        MAX(layer_flow_pressure) AS layer_flow_pressure
    FROM dp_rid_ods_corp_shtr_wellop.well_layer_op_inj
    WHERE layer_id = :pl_code
    GROUP BY well_id, layer_id, calc_year, calc_month
) AS wloi
  ON mer.sk_1 = wloi.well_id
 AND mer.pl_1 = wloi.layer_id
 AND CAST(mer.gg_1 AS INTEGER) = wloi.calc_year
 AND CAST(mer.dd_1 AS INTEGER) = wloi.calc_month
LEFT JOIN (
    SELECT uwi, object_name, MAX(node_x) AS node_x, MAX(node_y) AS node_y
    FROM dp_rid_ods_corp_geobd_super_man.plastcross
    WHERE orig_layers_name LIKE '%_T3'
      AND object_name IN (
          SELECT ne_1
          FROM dp_rid_ods_corp_oisproduction_db5555.class
          WHERE cd_1 = :pl_code
      )
    GROUP BY uwi, object_name
) AS pc
  ON g.uwi = pc.uwi
 AND pl.ne_1 = pc.object_name
WHERE mer.ms_1 = :ms_code
  AND mer.pl_1 = :pl_code
  -- Keep common EDW spellings explicit for consistent local/KSPD filtering.
  AND TRIM(sm.ns_1) IN ('РАБ.', 'РАБ', 'раб.', 'раб', 'WORK', 'work')
  AND CAST(mer.dd_1 AS INTEGER) BETWEEN 1 AND 12
  AND CAST(mer.gg_1 AS INTEGER) > 1900
ORDER BY _year DESC, _month DESC
"""


FRACS_SQL = """
SELECT
    CAST(g.smart_name AS TEXT) AS well_number,
    gi.date_op AS date,
    gi.layer AS object,
    CAST(COALESCE(gi.kpr, '0') AS REAL) AS permeability,
    CAST(COALESCE(gi.ppl, '0') AS REAL) AS "P_reservoir",
    CAST(COALESCE(gi.xf, '0') AS REAL) AS xfr,
    CAST(COALESCE(gi.wf, '0') AS REAL) AS w_f,
    CAST(COALESCE(gi.skin, '0') AS REAL) AS skin,
    CAST(COALESCE(gi.top, '0') AS REAL) AS perf_upper,
    CAST(COALESCE(gi.bot, '0') AS REAL) AS perf_lower,
    COALESCE(NULLIF(gi.comments, ''), CAST(gi.num_oper AS TEXT), '') AS comment
FROM dp_rid_ods_corp_geobd_super_man.grp_info AS gi
LEFT JOIN dp_rid_ods_corp_geobd_dwh.geooistranslation AS g
  ON gi.uwi = g.uwi
WHERE EXISTS (
    SELECT 1
    FROM dp_rid_ods_corp_oisproduction_db5555.class AS pl
    WHERE pl.cd_1 = :pl_code
      AND LOWER(TRIM(gi.layer)) = LOWER(TRIM(pl.ne_1))
)
AND EXISTS (
    SELECT 1
    FROM dp_rid_ods_corp_oisproduction_db5555.arx AS mer
    WHERE mer.sk_1 = g.sk_1
      AND mer.ms_1 = :ms_code
      AND mer.pl_1 = :pl_code
)
ORDER BY gi.date_op DESC
"""


PVT_SQL = """
SELECT p.*
FROM dp_rid_ods_corp_ngtsmart_rdfsmart.pvt AS p
WHERE EXISTS (
    SELECT 1
    FROM dp_rid_ods_corp_oisproduction_db5555.class AS ms
    JOIN dp_rid_ods_corp_oisproduction_db5555.class AS pl
      ON pl.cd_1 = :pl_code
    WHERE ms.cd_1 = :ms_code
      AND LOWER(TRIM(p.oil_field)) = LOWER(TRIM(ms.ne_1))
      AND LOWER(TRIM(p.layer)) = LOWER(TRIM(pl.ne_1))
)
"""


GRIDS_SQL = """
SELECT gridkey, name, gridtype, dt, field, object
FROM dp_rid_ods_corp_eragrad_api.gridlist AS grid
WHERE EXISTS (
    SELECT 1
    FROM dp_rid_ods_corp_oisproduction_db5555.class AS ms
    JOIN dp_rid_ods_corp_oisproduction_db5555.class AS pl
      ON pl.cd_1 = :pl_code
    WHERE ms.cd_1 = :ms_code
      AND LOWER(TRIM(grid.field)) = LOWER(TRIM(ms.ne_1))
      AND LOWER(TRIM(grid.object)) = LOWER(TRIM(pl.ne_1))
)
ORDER BY dt DESC NULLS LAST
"""


GRID_BY_KEY_SQL = """
SELECT *
FROM dp_rid_ods_corp_eragrad_api.gridlist
WHERE CAST(gridkey AS TEXT) = CAST(:gridkey AS TEXT)
LIMIT 1
"""
