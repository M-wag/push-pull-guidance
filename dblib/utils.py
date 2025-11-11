from dataclasses import dataclass, replace, asdict
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import sqlite3

#----------------------------------------------------------------------------
# Dataclasses modeling the possible values for experiments and parameters.

@dataclass(frozen=True)
class Parameter:
    name: str
    value : Any
    value_type : str

    def to_storage(self):
        value_stored, type_stored = value_to_storage(self.value)
        return (self.name, value_stored, type_stored)

@dataclass(frozen=True)
class ParameterSet:
    parameters: Tuple[Parameter, ...]

    def to_storage(self):
        return [parameter.to_storage() for parameter in self.parameters] 


@dataclass(frozen=True)
class Experiment:
    id: Optional[int]
    time_created : datetime
    lib_version : str
    experiment_name : str
    result_path : str
    samplers_prms : ParameterSet
    dynamics_prms : ParameterSet
    states_prms : ParameterSet
    notes : str

    def main_row(self):
        return (
            self.lib_version,
            self.experiment_name,
            self.result_path,
            self.notes,
        )

    def parameter_rows(self):
        tables = [table for table in asdict(self).keys() if table.endswith("_prms")]
        rows_per_table = {}
        for table in tables:
            table_name = table.split("_prms")[0]
            rows_per_table[table_name] = getattr(self, table).to_storage()
        return rows_per_table





    

#----------------------------------------------------------------------------
# Functions for handling the Experiment, Parameter and ParameterSet

def infer_value_type(value: Any) -> str:
    return {
        int : "number",
        float : "number", 
        bool : "boolean",
        str : "string", 
    }.get(type(value), "string")


def value_to_storage(value: Any) -> Tuple[str, str]:
    value_type = infer_value_type(value)
    storage_value = (
        str(value).lower() if value_type == 'boolean' 
        else str(value)
    )
    return (storage_value, value_type)


def dict_to_parameter_set(dict_params: Dict[str, Any]) -> ParameterSet:
    parameters = tuple(Parameter(name, value, infer_value_type(value)) 
        for name, value in dict_params.items())
    return ParameterSet(parameters)


def create_experiment(
        lib_version     :str,
        experiment_name :str,
        result_path     :str,
        samplers_prms   :Dict[str, Any],
        dynamics_prms   :Dict[str, Any],
        states_prms     :Dict[str, Any],
        notes           :str = ""

):
    return Experiment(
            id=None,
            time_created=datetime.now(),
            lib_version=lib_version,
            experiment_name=experiment_name,
            result_path=result_path,
            notes=notes,
            samplers_prms=dict_to_parameter_set(samplers_prms),
            dynamics_prms=dict_to_parameter_set(dynamics_prms),
            states_prms=dict_to_parameter_set(states_prms),
    )

def save_experiment(cursor: sqlite3.Cursor, experiment: Experiment) -> Experiment:
    # Insert main experiment
    cursor.execute("INSERT INTO experiments (lib_version, experiment_name, result_path, notes) VALUES (?, ?, ?, ?)", experiment.main_row())
    experiment_id = cursor.lastrowid
    # Insert parameter tables
    for table, parameters in experiment.parameter_rows().items():
        values = [(experiment_id, *parameter) for parameter in parameters]
        cursor.executemany(f"INSERT INTO {table} (experiment_id, parameter_name, parameter_value, value_type) VALUES (?, ?, ?, ?)", values)
    # Return Experiment with experiment_id
    return replace(experiment, id=experiment_id)


#---------------------------------------------------------------------------
# Initialize a database that contains tables for:
# experiments, samplers, dynamics and states

def intialize_database(cursor: sqlite3.Cursor):
    schema_experiment = \
            '''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                lib_version TEXT NOT NULL,
                experiment_name TEXT NOT NULL,
                result_path TEXT NOT NULL,
                notes TEXT
            )
            '''
    schema_parameter = \
            '''
            CREATE TABLE IF NOT EXISTS {0} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                parameter_name TEXT NOT NULL,
                parameter_value TEXT NOT NULL,
                value_type TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE,
                UNIQUE(experiment_id, parameter_name)
            )
            '''


    names_table = ["samplers", "dynamics", "states"]
    schema = [schema_experiment] + [schema_parameter.format(name) for name in names_table]
    for statement in schema:
        cursor.execute(statement)

    
