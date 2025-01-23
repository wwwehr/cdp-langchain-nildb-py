from collections import defaultdict
import json
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
import os
from pydantic import BaseModel, Field
import requests
from typing import Union, Dict, List, Optional, Type
import uuid
import nilql

"""
import logging
import http.client

# Enable debug logging for http.client
http.client.HTTPConnection.debuglevel = 1

# Configure logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True
"""

JSON_TYPE = Dict[str, Union[str, int, float, bool, None, List, Dict]]

with open(os.environ.get("NILDB_CONFIG", ".nildb.config.json")) as fh:
    CONFIG = json.load(fh)


class NilDBAPI:
    def __init__(self):
        self.nodes = CONFIG["hosts"]
        self.secret_key = nilql.SecretKey.generate(
            {"nodes": [{}] * len(CONFIG["hosts"])}, {"store": True}
        )

    def data_download(self) -> Dict:
        """Download all records in the specified node and schema."""
        try:
            shares = defaultdict(list)
            teams = defaultdict(list)
            for idx, node in enumerate(self.nodes):
                headers = {
                    "Authorization": f'Bearer {node["bearer"]}',
                    "Content-Type": "application/json",
                }

                body = {"schema": CONFIG["schema_id"], "filter": {}}

                response = requests.post(
                    f"https://{node['url']}/api/v1/data/read",
                    headers=headers,
                    json=body,
                )
                assert (
                    response.status_code == 200
                ), ("upload failed: " + response.content)
                data = response.json().get("data")
                for i, d in enumerate(data):
                    shares[i].append(d["text"]["$share"])
                    teams[i].append(d["team"])
            for i in range(len(teams)):
                assert(teams[i][0] == teams[i][j] for j in range(1, len(teams[i])))
                teams[i] = teams[i][0]
            decrypted = []
            for k in shares:
                decrypted.append(nilql.decrypt(self.secret_key, shares[k]))
            messages = {}
            for team, message in zip(teams, decrypted):
                if len(messages) == 2:
                    break
                if teams[team] == "blue":
                    messages["blue"] = message
                elif teams[team] == "red":
                    messages["red"] = message
            return messages
        except Exception as e:
            print(f"Error retrieving records in node {idx}: {str(e)}")
            return {}


    def data_upload(self, payload: JSON_TYPE) -> bool:
        """Create/upload records in the specified node and schema."""
        try:
            print(json.dumps(payload))
            payload["text"] = {
                "$allot": nilql.encrypt(self.secret_key, payload["text"])
            }
            payloads = nilql.allot(payload)
            for idx, shard in enumerate(payloads):
                node = self.nodes[idx]
                headers = {
                    "Authorization": f'Bearer {node["bearer"]}',
                    "Content-Type": "application/json",
                }

                body = {"schema": CONFIG["schema_id"], "data": [shard]}

                response = requests.post(
                    f"https://{node['url']}/api/v1/data/create",
                    headers=headers,
                    json=body,
                )

                assert (
                    response.status_code == 200
                    and response.json().get("data", {}).get("errors", []) == []
                ), ("upload failed: " + response.content)
            return True
        except Exception as e:
            print(f"Error creating records in node {idx}: {str(e)}")
            return False

class NilDbUploadInput(BaseModel):
    text: str = Field(description="value to store")


class NilDbUploadTool(BaseTool):
    name: str = "nildb_upload_tool"
    description: str = """In addition, you can upload data into a privacy preserving database using the nildb_upload_tool"""
    args_schema: Type[BaseModel] = NilDbUploadInput
    return_direct: bool = True

    def _run(
        self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        nildb = NilDBAPI()
        my_id = str(uuid.uuid4())

        return (
            "ok"
            if nildb.data_upload(
                payload={"_id": my_id, "team": CONFIG["team"], "text": text}
            )
            else "nok"
        )


class NilDbDownloadInput(BaseModel):
    pass

class NilDbDownload(BaseModel):
    text: str = Field(description="value to fetch")


class NilDbDownloadTool(BaseTool):
    name: str = "nildb_download_tool"
    description: str = """In addition, you can download all data from a privacy preserving database using the nildb_download_tool"""
    args_schema: Type[BaseModel] = NilDbDownloadInput
    return_direct: bool = True

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict:

        nildb = NilDBAPI()

        return nildb.data_download()

