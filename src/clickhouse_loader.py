import clickhouse_connect

from src.config.settings import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
)


def get_clickhouse_client():
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE
    )
    return client

def create_table_if_not_exists():
    client = get_clickhouse_client()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS releases (
        date Date,
        event_name String,
        product_area String,
        teams_or_people Array(String),
        key_updates Array(String),
        impact String,
        additional_notes String
    )
    ENGINE = MergeTree
    ORDER BY date
    """
    client.command(create_table_query)

def insert_release_data(parsed_data: dict):
    """
    Insert one release record into ClickHouse.
    `parsed_data` is expected to have keys matching the table schema.
    """
    client = get_clickhouse_client()

    # Convert date string to a standard date format if needed
    date_str = parsed_data.get("date") or "1970-01-01"
    event_name = parsed_data.get("event_name") or ""
    product_area = parsed_data.get("product_area") or ""
    teams_or_people = parsed_data.get("teams_or_people") or []
    key_updates = parsed_data.get("key_updates") or []
    impact = parsed_data.get("impact") or ""
    additional_notes = parsed_data.get("additional_notes") or ""

    # Insert query
        date,
        event_name,
        product_area,
        teams_or_people,
        key_updates,
        impact,
        additional_notes
    ) VALUES
    """
    # Insert one row (array fields must be lists, etc.)
    client.insert(
        "releases",
        [[
            date_str,
            event_name,
            product_area,
            teams_or_people,
            key_updates,
            impact,
            additional_notes
        ]],
        column_names=[
            "date",
            "event_name",
            "product_area",
            "teams_or_people",
            "key_updates",
            "impact",
            "additional_notes"
        ]
    )
