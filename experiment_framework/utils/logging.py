"""This module provides logging utilities for the experiment framework.

"""

import os
import time
import dataclasses
import json
import pyarrow as pa
import pyarrow.parquet as pq


def init_experiment_folder(data_folder: str, experiment_name: str = "data", timed: bool = True) -> str:
    """This function initializes the experiment folder.

    Args:
        experiment_name (str): The name of the experiment.
        data_folder (str): The folder to store the experiment data.
    """
    # create data folder
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # create experiment folder
    if timed:
        experiment_folder_name = experiment_name + "_" + time.strftime("%Y%m%d-%H%M%S")
    else:
        experiment_folder_name = experiment_name
    experiment_folder = os.path.join(data_folder, experiment_folder_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    else:
        # raise an warning if the folder already exists
        print(f"Warning: {experiment_folder} already exists.")

    return experiment_folder

def dataclass_to_pyarrow_schema(dataclass: dataclasses.dataclass) -> pa.Schema:
    """Converts a dataclass to a pyarrow schema.

    Args:
        dataclass (dataclass): The dataclass to convert.

    Returns:
        pa.Schema: The pyarrow schema.
    """
    pa_fields = []
    for field in dataclasses.fields(dataclass):
        field_name = field.name
        field_type = field.type
        pa_field = pa.field(field_name, field_type())
        pa_fields.append(pa_field)
    return pa.schema(pa_fields)

def save_dataclass_to_json(dataclass: dataclasses.dataclass, filepath: str, encoding: str = "utf-8"):
    """Saves a dataclass to a json file.

    Args:
        dataclass (dataclass): The dataclass to save.
        filepath (str): The path to save the dataclass.
    """
    data = dataclasses.asdict(dataclass)
    with open(filepath, "w", encoding=encoding) as f:
        json.dump(data, f)


# Rewriting the class again with the necessary imports
class ParquetTableRecorder:
    """This class is used to record experiment results in parquet format."""

    def __init__(self, filepath: str, schema: pa.Schema, batch_size: int = 100, overwrite: bool = True):
        """
        Initializes a new parquet file for recording.

        Args:
            filepath (str): The path to the parquet file to be created.
            schema (pa.Schema): The schema for the data to be written.
            batch_size (int): The number of rows to batch before writing to parquet.
        """
        self.filepath = filepath
        self.schema = schema
        self.batch_size = batch_size
        self.batch = []  # Temporary storage for the incoming records
        self.writer = None  # Parquet writer
        self.recording = False
        self.overwrite = overwrite

        # Initialize a new parquet file
        self._initialize_parquet_file()

    def _initialize_parquet_file(self):
        """Initializes the parquet file for writing with the given schema."""
        # if file already exists, delete it and print a warning
        if os.path.exists(self.filepath):
            print(f"Warning: {self.filepath} already exists.")
            if self.overwrite:
                print(f"Overwriting {self.filepath}")
                os.remove(self.filepath)
            else:
                # check if the file is open
                if self.writer:
                    raise ValueError("The file is already open. Close the file before overwriting.")
                
                # check schema
                existing_schema = pq.read_table(self.filepath).schema
                if existing_schema != self.schema:
                    raise ValueError("The schema of the existing file does not match the given schema.")
                
                print("Appending to the existing file.")
        self.writer = pq.ParquetWriter(self.filepath, self.schema)
        self.recording = True

    def record(self, record_data: dict):
        """
        Records new data into the parquet file.

        Args:
            record_data (dict): The data to record. The keys should match the schema columns.
        """
        # Convert the record data to a pyarrow table
        table = pa.Table.from_pydict(record_data, schema=self.schema)

        # Add new data to the current batch
        self.batch.append(table)

        # Check if the batch size has been exceeded
        if len(self.batch) >= self.batch_size:
            self._write_batch()
            self.batch = []  # Clear the batch

    def _write_batch(self):
        """Writes the current batch of data to the parquet file."""
        # Concatenate all tables in the batch and write to parquet
        combined_batch = pa.concat_tables(self.batch)
        self.writer.write_table(combined_batch)
        print(f"Batch written to {self.filepath}")

    def close(self):
        """Finalizes the parquet file by writing any remaining data and closing the file."""
        if self.batch:
            self._write_batch()  # Write any remaining records
        if self.writer:
            self.writer.close()
            self.recording = False
