from pdb import set_trace as bp
from collections import defaultdict
import json
from langchain.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
import os
from pydantic import BaseModel, Field
import requests
from typing import Union, Dict, List, Optional, Type
import uuid

import jwt
import time
from ecdsa import SigningKey, SECP256k1


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
LLM_TYPE = Union[BaseLLM, BaseChatModel]

with open(os.environ.get("NILLION_DB_CONFIG_FILEPATH", ".nildb.config.json")) as fh:
    CONFIG = json.load(fh)


class BaseToolWithLlm(BaseTool):
    llm: LLM_TYPE = None

    def __init__(self, llm: LLM_TYPE):
        super().__init__()
        self.llm = llm


class NilDBAPI:
    llm: LLM_TYPE = None

    def __init__(self, llm: LLM_TYPE = None):
        self.nodes = CONFIG["hosts"]
        self.init_jwt()
        self.secret_key = nilql.SecretKey.generate(
            {"nodes": [{}] * len(CONFIG["hosts"])}, {"store": True}
        )
        self.llm = llm

    def init_jwt(self) -> bool:
        """
        Create JWTs signed with ES256K for multiple node_ids
        """

        # Convert the secret key from hex to bytes
        private_key = bytes.fromhex(CONFIG["secret_key"])
        signer = SigningKey.from_string(private_key, curve=SECP256k1)

        for node in self.nodes:
            # Create payload for each node_id
            payload = {
                "iss": CONFIG["owner_id"],
                "aud": node["name"],
                "exp": int(time.time()) + 3600,
            }

            # Create and sign the JWT
            node["bearer"] = jwt.encode(payload, signer.to_pem(), algorithm="ES256K")

        return True

    def lookup_schema(self, schema_description: str) -> str:
        """Lookup a JSON schema based on input description and return it's UUID"""
        print(f"fn:lookup_schema [{schema_description}]")
        try:

            headers = {
                "Authorization": f'Bearer {self.nodes[0]["bearer"]}',
                "Content-Type": "application/json",
            }

            response = requests.get(
                f"https://{self.nodes[0]['url']}/api/v1/schemas", headers=headers
            )

            assert (
                response.status_code == 200 and response.json().get("errors", []) == []
            ), response.content.decode("utf8")

            schema_prompt = f"""
            1. I'll provide you with a description of the schema I want to use
            2. I'll provide you with a list of available schemas
            3. You will select the best match and return the associated UUID from the outermost `_id` field
            4. Do not include explanation or comments. Only a valid UUID string
            5. Based on the provided description, select a schema from the provided schemas.

            DESIRED SCHEMA DESCRIPTION:
            {schema_description}

            AVAILABLE SCHEMAS:
            {response.text}
            """

            response = self.llm.invoke(schema_prompt)

            print(response.content)
            return response.content

        except Exception as e:
            print(f"Error creating schema: {str(e)}")
            return False

    def create_schema(self, schema_description: str) -> bool:
        """Creates a JSON schema based on input description and uploads it to nildb"""
        print(f"fn:create_schema [{schema_description}]")

        try:

            schema_prompt = f"""
            1. I'll provide you with a description of the schema I want to implement
            3. For any fields that could be considered financial, secret, currency, value holding, political, family values, sexual, criminal, risky, personal, private or personally 
               identifying (PII), I want you to replace that type and value, instead, with an object that has a key named `$share` and the value of string as shown in this example:

                ORIGINAL ATTRIBUTE:
                "password": {{
                  "type": "string"
                }}

                REPLACED WITH UPDATED ATTRIBUTE PRESERVING NAME:
                "password": {{
                    "type": "object",
                    "properties": {{
                        "$share": {{
                          "type": "string",
                         }}
                     }}
                }}
            
            4. The JSON document should follow the patterns shown in these examples contained herein where the final result is ready to be included in the POST JSON payload
            5. Do not include explanation or comments. Only a valid JSON payload document.
            
            START OF JSON SCHEMA DESECRIPTION
            
            a JSON Schema following these requirements:
            
            - Use JSON Schema draft-07, type "array"
            - Each record needs a unique \_id (UUID format, coerce: true)
            - Use "date-time" format for dates (coerce: true)
            - Mark required fields (\_id is always required)
            - Set additionalProperties to false
            - Avoid "$" prefix in field names to prevent query conflicts
            - The schema to create is embedded in the "schema" attribute
            - "_id" should be the only "keys"
            - Note: System adds \_created and \_updated fields automatically
            
            Example `POST /schema` Payload
            
            {{
              "name": "My services",
              "keys": ["_id"],
              "schema": []{{
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "array",
                "items": {{
                  "type": "object",
                  "properties": {{
                    "_id": {{
                      "type": "string",
                      "format": "uuid",
                      "coerce": true
                    }},
                    "username": {{
                      "type": "string"
                    }},
                    "password": {{
                      "type": "string"
                    }},
                  }},
                  "required": ["_id", "username", "password"],
                  "additionalProperties": false
                }}
              }}
            }}
            
            Based on this description, create a JSON schema:
            {schema_description}
            """
            response = self.llm.invoke(schema_prompt)

            schema = json.loads(response.content)

            schema["_id"] = str(uuid.uuid4())
            schema["owner"] = CONFIG["owner_id"]

            print(json.dumps(schema, indent=4))

            for node in self.nodes:
                headers = {
                    "Authorization": f'Bearer {node["bearer"]}',
                    "Content-Type": "application/json",
                }

                response = requests.post(
                    f"https://{node['url']}/api/v1/schemas",
                    headers=headers,
                    json=schema,
                )

                assert (
                    response.status_code == 200
                    and response.json().get("errors", []) == []
                ), response.content.decode("utf8")
            return True
        except Exception as e:
            print(f"Error creating schema: {str(e)}")
            return False

    def data_download(self, schema_id: str) -> Dict:
        """Download all records in the specified node and schema."""
        print(f"fn:data_download [{schema_id}]")
        try:
            shares = defaultdict(list)
            teams = defaultdict(list)
            for idx, node in enumerate(self.nodes):
                headers = {
                    "Authorization": f'Bearer {node["bearer"]}',
                    "Content-Type": "application/json",
                }

                body = {
                    "schema": schema_id,
                    "filter": {"contest": CONFIG["contest"]},
                }

                response = requests.post(
                    f"https://{node['url']}/api/v1/data/read",
                    headers=headers,
                    json=body,
                )
                assert response.status_code == 200, "upload failed: " + response.content
                data = response.json().get("data")
                for i, d in enumerate(data):
                    shares[i].append(d["text"]["$share"])
                    teams[i].append(d["team"])
            for i in range(len(teams)):
                assert (teams[i][0] == teams[i][j] for j in range(1, len(teams[i])))
                teams[i] = teams[i][0]
            decrypted = []
            for k in shares:
                decrypted.append(nilql.decrypt(self.secret_key, shares[k]))
            messages = {}
            judged = {"blue": True, "purple": True, "red": True}
            for team, message in zip(teams, decrypted):
                if len(messages) == len(judged):
                    break
                if teams[team] in judged and judged[teams[team]]:
                    messages[teams[team]] = message
                    judged[teams[team]] = False
            return messages
        except Exception as e:
            print(f"Error retrieving records in node {idx}: {str(e)}")
            return {}

    def data_upload(self, schema_id: str, payload: JSON_TYPE) -> bool:
        """Create/upload records in the specified node and schema."""
        print(f"fn:data_upload [{schema_id}] [{payload}]")
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

                body = {"schema": schema_id, "data": [shard]}

                response = requests.post(
                    f"https://{node['url']}/api/v1/data/create",
                    headers=headers,
                    json=body,
                )

                assert (
                    response.status_code == 200
                    and response.json().get("errors", []) == []
                ), ("upload failed: " + response.content)
            return True
        except Exception as e:
            print(f"Error creating records in node {idx}: {str(e)}")
            return False


class NilDbSchemainput(BaseModel):
    schema_description: str = Field(
        description="a complete description of the desired nildb schema"
    )


class NilDbSchemaLookupTool(BaseToolWithLlm):
    name: str = "nildb_schema_lookup_tool"
    description: str = """In addition, you can lookup schemas in your privacy preserving database based on an input string using the nildb_schema_lookup_tool"""
    args_schema: Type[BaseModel] = NilDbSchemainput
    return_direct: bool = True

    def _run(
        self,
        schema_description: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        nildb = NilDBAPI(self.llm)

        return (
            "ok"
            if nildb.lookup_schema(schema_description=schema_description)
            else "nok"
        )


class NilDbSchemaCreateTool(BaseToolWithLlm):
    name: str = "nildb_schema_create_tool"
    description: str = """In addition, you can create schemas in your privacy preserving database based on an input string using the nildb_schema_create_tool"""
    args_schema: Type[BaseModel] = NilDbSchemainput
    return_direct: bool = True

    def _run(
        self,
        schema_description: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        nildb = NilDBAPI(self.llm)

        return (
            "ok"
            if nildb.create_schema(schema_description=schema_description)
            else "nok"
        )


class NilDbUploadInput(BaseModel):
    schema_id: str = Field(description="the UUID of the nildb schema")
    text: str = Field(description="value to store")


class NilDbUploadTool(BaseTool):
    name: str = "nildb_upload_tool"
    description: str = """In addition, you can upload data into a privacy preserving database using the nildb_upload_tool"""
    args_schema: Type[BaseModel] = NilDbUploadInput
    return_direct: bool = True

    def _run(
        self,
        schema_id: str,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        nildb = NilDBAPI()
        my_id = str(uuid.uuid4())

        return (
            "ok"
            if nildb.data_upload(
                schema_id=schema_id,
                payload={
                    "_id": my_id,
                    "contest": CONFIG["contest"],
                    "team": CONFIG["team"],
                    "text": text,
                },
            )
            else "nok"
        )


class NilDbDownloadInput(BaseModel):
    pass


class NilDbDownload(BaseModel):
    schema_id: str = Field(description="the UUID of the nildb schema")


class NilDbDownloadTool(BaseTool):
    name: str = "nildb_download_tool"
    description: str = """In addition, you can download all data from a privacy preserving database using the nildb_download_tool"""
    args_schema: Type[BaseModel] = NilDbDownloadInput
    return_direct: bool = True

    def _run(
        self, schema_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict:

        nildb = NilDBAPI()

        return nildb.data_download(schema_id)
