from pdb import set_trace as bp
from collections import defaultdict
from copy import deepcopy
import json
from jsonschema import validators, Draft7Validator
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
        self.init_jwt()
        print(f"initializing for secret sharing across {len(self.nodes)} nodes")
        self.cluster_key = nilql.ClusterKey.generate(
            {"nodes": [{}] * len(self.nodes)}, {"store": True}
        )
        self.llm = llm

    def init_jwt(self) -> bool:
        """
        Create JWTs signed with ES256K for multiple node_ids
        """

        response = requests.post(
            "https://sv-sda-registration.replit.app/api/config",
            headers={
                "Content-Type": "application/json",
            },
            json={"org_did": CONFIG["org_did"]},
        )
        self.nodes = response.json()["nodes"]

        # Convert the secret key from hex to bytes
        private_key = bytes.fromhex(CONFIG["secret_key"])
        signer = SigningKey.from_string(private_key, curve=SECP256k1)

        for node in self.nodes:
            # Create payload for each node_id
            payload = {
                "iss": CONFIG["org_did"],
                "aud": node["did"],
                "exp": int(time.time()) + 3600,
            }

            # Create and sign the JWT
            node["bearer"] = jwt.encode(payload, signer.to_pem(), algorithm="ES256K")

        return True

    def mutate_secret_attributes(self, entry: dict) -> None:
        keys = list(entry.keys())
        for key in keys:
            value = entry[key]
            if key == "$share":
                del entry["$share"]
                entry["$allot"] = nilql.encrypt(self.cluster_key, value)
            elif isinstance(value, dict):
                self.mutate_secret_attributes(value)

    def validator_builder(self):
        """builds a validator to validate the candidate document against loaded schema"""
        return validators.extend(Draft7Validator)

    def _filter_schemas(self, schema_uuid: str, schema_list: list) -> dict:
        my_schema = None
        for this_schema in schema_list:
            if this_schema["_id"] == schema_uuid:
                my_schema = this_schema["schema"]
                break
        assert my_schema is not None, "failed to lookup schema"
        return my_schema

    def lookup_schema(self, schema_description: str) -> tuple:
        """Lookup a JSON schema based on input description and return it's UUID"""
        print(f"fn:lookup_schema [{schema_description}]")
        try:

            headers = {
                "Authorization": f'Bearer {self.nodes[0]["bearer"]}',
                "Content-Type": "application/json",
            }

            response = requests.get(
                f"{self.nodes[0]['url']}/api/v1/schemas", headers=headers
            )

            assert (
                response.status_code == 200 and response.json().get("errors", []) == []
            ), response.content.decode("utf8")

            schema_list = response.json()["data"]
            assert len(schema_list) > 0, "failed to fetch schemas from nildb"

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

            my_uuid = response.content
            my_uuid = "".join(c for c in my_uuid if c.lower() in "0123456789abcdef-")

            my_schema = self._filter_schemas(my_uuid, schema_list)
            return my_uuid, my_schema

        except Exception as e:
            print(f"Error looking up schema: {str(e)}")
            return None

    def create_schema(self, schema_description: str) -> dict:
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
            schema["owner"] = CONFIG["org_did"]

            for node in self.nodes:
                headers = {
                    "Authorization": f'Bearer {node["bearer"]}',
                    "Content-Type": "application/json",
                }

                response = requests.post(
                    f"{node['url']}/api/v1/schemas",
                    headers=headers,
                    json=schema,
                )

                assert (
                    response.status_code == 200
                    and response.json().get("errors", []) == []
                ), response.content.decode("utf8")
            return schema
        except Exception as e:
            print(f"Error creating schema: {str(e)}")
            return ""

    def data_download(self, schema_id: str) -> Dict:
        """Download all records in the specified node and schema."""
        print(f"fn:data_download [{schema_id}]")
        try:
            shares = defaultdict(list)
            for idx, node in enumerate(self.nodes):
                headers = {
                    "Authorization": f'Bearer {node["bearer"]}',
                    "Content-Type": "application/json",
                }

                body = {
                    "schema": schema_id,
                    "filter": {},
                }

                response = requests.post(
                    f"{node['url']}/api/v1/data/read",
                    headers=headers,
                    json=body,
                )
                assert (
                    response.status_code == 200
                ), "upload failed: " + response.content.decode("utf8")
                data = response.json().get("data")
                for i, d in enumerate(data):
                    shares[d["_id"]].append(d)
            decrypted = []
            for k in shares.keys():
                decrypted.append(nilql.unify(self.cluster_key, shares[k]))
            return decrypted
        except Exception as e:
            print(f"Error retrieving records in node: {str(e)}")
            return {}

    def data_upload(self, schema_uuid: str, payload: list) -> bool:
        """Create/upload records in the specified node and schema."""
        print(f"fn:data_upload [{schema_uuid}] [{payload}]")
        try:

            headers = {
                "Authorization": f'Bearer {self.nodes[0]["bearer"]}',
                "Content-Type": "application/json",
            }

            response = requests.get(
                f"{self.nodes[0]['url']}/api/v1/schemas", headers=headers
            )

            my_schema = self._filter_schemas(schema_uuid, response.json()["data"])

            builder = self.validator_builder()
            validator = builder(my_schema)

            for entry in payload:
                self.mutate_secret_attributes(entry)

            payloads = nilql.allot(payload)

            for idx, shard in enumerate(payloads):

                validator.validate(shard)

                node = self.nodes[idx]
                headers = {
                    "Authorization": f'Bearer {node["bearer"]}',
                    "Content-Type": "application/json",
                }

                body = {"schema": schema_uuid, "data": shard}

                response = requests.post(
                    f"{node['url']}/api/v1/data/create",
                    headers=headers,
                    json=body,
                )

                assert (
                    response.status_code == 200
                    and response.json().get("errors", []) == []
                ), "upload failed: " + response.content.decode("utf8")
            return True
        except Exception as e:
            print(f"Error creating records in node: {str(e)}")
            return False


class NilDbSchemainput(BaseModel):
    schema_description: str = Field(
        description="a complete description of the desired nildb schema"
    )


class NilDbSchemaLookupTool(BaseToolWithLlm):
    name: str = "nildb_schema_lookup_tool"
    description: str = """In addition, you can lookup schemas in your privacy preserving database based on an input string using the nildb_schema_lookup_tool. Remember the UUID and schema values for future use with nildb_upload_tool and nildb_download_tool"""
    args_schema: Type[BaseModel] = NilDbSchemainput
    return_direct: bool = False

    def _run(
        self,
        schema_description: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> tuple:
        nildb = NilDBAPI(self.llm)

        return nildb.lookup_schema(schema_description=schema_description)


class NilDbSchemaCreateTool(BaseToolWithLlm):
    name: str = "nildb_schema_create_tool"
    description: str = """In addition, you can create schemas in your privacy preserving database based on an input string using the nildb_schema_create_tool"""
    args_schema: Type[BaseModel] = NilDbSchemainput
    return_direct: bool = False

    def _run(
        self,
        schema_description: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        nildb = NilDBAPI(self.llm)

        return nildb.create_schema(schema_description=schema_description)


class NilDbUploadInput(BaseModel):
    schema_uuid: str = Field(
        description="the UUID obtained from the nildb_schema_lookup_tool"
    )
    data_to_store: list = Field(
        description="data to store in nildb that matches schema"
    )


class NilDbUploadTool(BaseTool):
    name: str = "nildb_upload_tool"
    description: str = """In addition, you can upload data into a privacy preserving database using the nildb_upload_tool. The _id field should always be a UUID4."""
    args_schema: Type[BaseModel] = NilDbUploadInput
    return_direct: bool = True

    def _run(
        self,
        schema_uuid: str,
        data_to_store: list,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:

        nildb = NilDBAPI()

        return (
            "ok"
            if nildb.data_upload(schema_uuid=schema_uuid, payload=data_to_store)
            else "nok"
        )


class NilDbDownloadInput(BaseModel):
    schema_uuid: str = Field(
        description="the UUID obtained from the nildb_schema_lookup_tool"
    )


class NilDbDownloadTool(BaseTool):
    name: str = "nildb_download_tool"
    description: str = """In addition, you can download all data from a privacy preserving database using the nildb_download_tool"""
    args_schema: Type[BaseModel] = NilDbDownloadInput
    return_direct: bool = True

    def _run(
        self,
        schema_uuid: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict:

        nildb = NilDBAPI()

        return nildb.data_download(schema_uuid)
