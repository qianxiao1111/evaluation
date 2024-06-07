# this file contains all of the helper functions used for evaluations

import itertools
import re
from func_timeout import func_timeout
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sqlalchemy import create_engine, text
import time

LIKE_PATTERN = r"LIKE[\s\S]*'"


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes a dataframe by:
    1. removing all duplicate rows
    2. sorting columns in alphabetical order
    3. sorting rows using values from first column to last (if query_category is not 'order_by' and question does not ask for ordering)
    4. resetting index
    """
    # remove duplicate rows, if any
    df = df.drop_duplicates()

    # sort columns in alphabetical order of column names
    sorted_df = df.reindex(sorted(df.columns), axis=1)
    sorted_df = sorted_df.sort_values(by=list(sorted_df.columns))
    # reset index
    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df


# extrapolate all possible queries from a query with { } in it
def get_all_minimal_queries(query: str) -> "list[str]":
    """
    extrapolate all possible queries
    - split by semicolon. this is to accommodate queries where joins to other tables are also acceptable.
    - expand all column permutations if there are braces { } in it. eg:
    ```sql
        SELECT {user.id, user.name} FROM user;
    ```
    Would be expanded to:
    ```sql
        SELECT user.id FROM user;
        SELECT user.name FROM user;
        SELECT user.id, user.name FROM user;
    ```
    """
    queries = query.split(";")
    result_queries = []
    for query in queries:
        query = query.strip()
        if query == "":
            continue
        start, end = find_bracket_indices(query, 0)
        if (start, end) == (-1, -1):
            result_queries.append(query)
            continue
        else:
            # get all possible column subsets
            column_options = query[start + 1 : end].split(",")
            column_combinations = list(
                itertools.chain.from_iterable(
                    itertools.combinations(column_options, r)
                    for r in range(1, len(column_options) + 1)
                )
            )
            for column_tuple in column_combinations:
                left = query[:start]
                column_str = ", ".join(column_tuple)
                right = query[end + 1 :]
                # change group by size dynamically if necessary
                if right.find("GROUP BY {}"):
                    right = right.replace("GROUP BY {}", f"GROUP BY {column_str}")
                result_queries.append(left + column_str + right)
    return result_queries


def compare_df(
    df_gold: pd.DataFrame,
    df_gen: pd.DataFrame,
) -> bool:
    """
    Compares two dataframes and returns True if they are the same, else False.
    query_gold and query_gen are the original queries that generated the respective dataframes.
    """
    # drop duplicates to ensure equivalence
    try:
        is_equal = df_gold.values == df_gen.values
        if is_equal.all():
            return True
    except:
        try:
            is_equal = df_gold.values == df_gen.values
            if is_equal:
                return True
        except:
            pass

    df_gold = normalize_table(df_gold)
    df_gen = normalize_table(df_gen)

    # perform same checks again for normalized tables
    if df_gold.shape != df_gen.shape:
        return False
    # fill NaNs with -99999 to handle NaNs in the dataframes for comparison
    df_gen.fillna(-99999, inplace=True)
    df_gold.fillna(-99999, inplace=True)
    is_equal = df_gold.values == df_gen.values
    try:
        return is_equal.all()
    except:
        return is_equal


def subset_df(
    df_sub: pd.DataFrame,
    df_super: pd.DataFrame,
    verbose: bool = False,
) -> bool:
    """
    Checks if df_sub is a subset of df_super.
    """
    if df_sub.empty:
        return False  # handle cases for empty dataframes

    # make a copy of df_super so we don't modify the original while keeping track of matches
    df_super_temp = df_super.copy(deep=True)
    matched_columns = []
    for col_sub_name in df_sub.columns:
        col_match = False
        for col_super_name in df_super_temp.columns:
            col_sub = df_sub[col_sub_name].sort_values().reset_index(drop=True)
            col_super = (
                df_super_temp[col_super_name].sort_values().reset_index(drop=True)
            )

            try:
                assert_series_equal(
                    col_sub, col_super, check_dtype=False, check_names=False
                )
                col_match = True
                matched_columns.append(col_super_name)
                # remove col_super_name to prevent us from matching it again
                df_super_temp = df_super_temp.drop(columns=[col_super_name])
                break
            except AssertionError:
                continue

        if not col_match:
            if verbose:
                print(f"no match for {col_sub_name}")
            return False

    df_sub_normalized = normalize_table(df_sub)

    # get matched columns from df_super, and rename them with columns from df_sub, then normalize
    df_super_matched = df_super[matched_columns].rename(
        columns=dict(zip(matched_columns, df_sub.columns))
    )
    df_super_matched = normalize_table(df_super_matched)

    try:
        assert_frame_equal(df_sub_normalized, df_super_matched, check_dtype=False)
        return True
    except AssertionError:
        return False
