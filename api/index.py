from fastapi import FastAPI

app = FastAPI(debug=True)


@app.get("/api/python")
def hello_world():
    return {"message": "Hello World"}


from fastapi import File, UploadFile
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import base64
from pydantic import BaseModel


import os

PAT = os.getenv("PAT")
USER_ID = "clarifai"
APP_ID = "main"
MODEL_ID = "age-demographics-recognition"
MODEL_VERSION_ID = "fb9f10339ac14e23b8e960e74984401b"


class Image(BaseModel):
    data: str


@app.post("/api/age")
async def analyze_age(image: Image):
    base64_encoded_image = image.data.split(",")[1]
    base64_bytes = base64.b64decode(base64_encoded_image)

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (("authorization", "Key " + PAT),)
    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(base64=base64_bytes)
                    )
                )
            ],
        ),
        metadata=metadata,
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        raise Exception(
            "Post model outputs failed, status: "
            + post_model_outputs_response.status.description
        )

    output = post_model_outputs_response.outputs[0]
    print("Predicted concepts:" + str(output.data.concepts))
    concepts = [
        {"name": concept.name, "value": concept.value}
        for concept in output.data.concepts
    ]
    return {"concepts": concepts}
