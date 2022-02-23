import pandas as pd
from copy import deepcopy


def apply(all_df, return_obj_dataframe=False, parameters=None):
    if parameters is None:
        parameters = {}

    eve_cols = [x for x in all_df.columns if not x.startswith("object_")]
    obj_cols = [x for x in all_df.columns if x.startswith("object_")]
    df = all_df[eve_cols]
    obj_df = pd.DataFrame()
    if obj_cols:
        obj_df = all_df[obj_cols]
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    if "event_start_timestamp" in df.columns:
        df["event_start_timestamp"] = pd.to_datetime(
            df["event_start_timestamp"])
    df = df.dropna(subset=["event_id"])
    df["event_id"] = df["event_id"].astype(str)
    df.type = "succint"

    if return_obj_dataframe:
        obj_df = obj_df.dropna(subset=["object_id"])
        return df, obj_df

    return df


def filter_by_timestamp(df, start_timestamp=None, end_timestamp=None):
    if start_timestamp is not None:
        df = df.loc[df["event_timestamp"] >= start_timestamp]
    if end_timestamp is not None:
        df = df.loc[df["event_timestamp"] <= end_timestamp]
    return df


def filter_object_df_by_object_ids(df, ids):
    df = df.loc[df["object_id"].isin(ids)]
    return df



def succint_stream_to_exploded_stream(stream):
    new_stream = []

    for ev in stream:
        keys = set(ev.keys())

        event_keys = [k for k in keys if k.startswith("event_")]
        object_keys = [k for k in keys if not k in event_keys]

        basic_event = {k: ev[k] for k in event_keys}

        for k in object_keys:
            if type(ev[k]) is str:
                if ev[k][0] == "[":
                    ev[k] = eval(ev[k])
                    #ev[k] = ev[k][1:-1].split(",")
            values = ev[k]
            if values is not None:
            #if values is not None and len(values) > 0:
                if not (str(values).lower() == "nan" or str(values).lower() == "nat"):
                    for v in values:
                        event = deepcopy(basic_event)
                        event[k] = v
                        new_stream.append(event)

    return new_stream

def succint_mdl_to_exploded_mdl(df):
    stream = df.to_dict('r')

    exploded_stream = succint_stream_to_exploded_stream(stream)

    df = pd.DataFrame(exploded_stream)
    df.type = "exploded"

    return df

