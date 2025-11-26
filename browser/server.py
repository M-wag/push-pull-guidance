import os
from flask import Flask, request, send_file, jsonify
from tabulate import tabulate
import sqlite3


app = Flask(__name__)

def show(cur):
    rows = cur.fetchall()
    colnames = [d[0] for d in cur.description]
    print(tabulate(rows, headers=colnames, tablefmt="github"))

def get_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    return conn, cursor

@app.post("/get_experiment_paths")
def get_experiment_paths():
    data = request.get_json()
    experiment_name = data.get("experimentName", "")
    _, cur = get_db("../data/db/demos.db")
    query = """
        SELECT result_path
        FROM experiments as e
        WHERE e.experiment_name = ?
    """
    rows = cur.execute(query, (experiment_name,)).fetchall()
    result = [row[0] for row in rows]
    return jsonify(result)

@app.post("/get_slider_domain")
def get_slider_domain():

    slider_id = request.get_json()
    _, cur = get_db("../data/db/demos.db")
    cur.execute("""
        CREATE TEMP TABLE temp AS
        SELECT 
            d.experiment_id, 
            CASE WHEN s.parameter_name = "id_class" THEN s.parameter_value END as id_class,
            SUM(CASE WHEN d.parameter_name = "vector_field" THEN d.parameter_value END) as vector_field,
            CASE WHEN d.parameter_name = "maps" THEN json_extract(d.parameter_value, "$[2].n_features") END as n_features,
            CASE WHEN d.parameter_name = "maps" THEN json_extract(d.parameter_value, "$[2].dim_out") END as dim_out
        FROM dynamics as d 
        INNER JOIN states as s
            ON s.experiment_id = d.experiment_id
        GROUP BY d.experiment_id
    """)
    match slider_id:
        case "nu":
            cur.execute("SELECT DISTINCT vector_field FROM temp")
        case "numFeatures":
            cur.execute("SELECT DISTINCT n_features FROM temp")
        case "numDims":
            cur.execute("SELECT DISTINCT dim_out FROM temp")
    rows = cur.fetchall()
    result = [row[0] for row in rows]
    return jsonify(result)


@app.post("/get_experiment_urls")
def get_experiment_urls():
    data = request.get_json() or {}
    experiment_name = data.get("experimentName") or "%"
    id_example = data.get("exampleId") or "%"
    id_class = data.get("classId") or "%"
    n_features = data.get("numFeatures") or "%"
    dim_out = data.get("numDims") or "%"
    nu = data.get("nu") or "%"

    conn, cur = get_db("../data/db/demos.db")

    # create the temp table as before
    cur.execute("""
        CREATE TEMP TABLE temp AS
        SELECT 
            d.experiment_id, 
            CASE WHEN s.parameter_name = "id_class" THEN s.parameter_value END as id_class,
            SUM(CASE WHEN d.parameter_name = "vector_field" THEN d.parameter_value END) as vector_field,
            CASE WHEN d.parameter_name = "maps" THEN json_extract(d.parameter_value, "$[2].n_features") END as n_features,
            CASE WHEN d.parameter_name = "maps" THEN json_extract(d.parameter_value, "$[2].dim_out") END as dim_out
        FROM dynamics as d 
        INNER JOIN states as s
            ON s.experiment_id = d.experiment_id
        GROUP BY d.experiment_id
    """)

    cur.execute("SELECT * from experiments INNER JOIN temp on experiments.id = temp.experiment_id limit 10") 
    show(cur)
    # base query and params (keep LIKE "%" defaults)
    query = """
        SELECT result_path
        FROM experiments
        INNER JOIN temp
            ON experiments.id = temp.experiment_id
        WHERE 
            experiment_name LIKE ?
            AND id_class LIKE ?
            AND dim_out LIKE ? 
            AND n_features LIKE ?
    """
    params = [experiment_name, id_class, dim_out, n_features]

    # Only add the vector_field filter if nu was provided (not "%")
    if nu != "%":
        nu_val = float(nu)
        tolerance = 1e-2
        query += " AND ABS(CAST(vector_field AS REAL) - ?) <= ?"
        params.extend([nu_val, tolerance])

    # If you want to restore filtering on n_features / dim_out later, add similar checks here.
    rows = cur.execute(query, tuple(params)).fetchall()

    result = ["/image_by_path?path=../" + row[0] for row in rows]
    return jsonify(result)

@app.get("/image_by_path")
def image_by_path():
    path = request.args.get("path")
    if not os.path.isfile(path):
        return flask.abort(404)
    return send_file(path)

@app.get("/")
def index():
    return send_file("index.html")

