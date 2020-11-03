from concurrent import futures
import grpc
import grpc_pb2
import grpc_pb2_grpc
import Unet
import pickle
import numpy as np
from tensorflow.keras.models import model_from_json

model = Unet.Unet()
#python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. grpc.proto

whole_size = 0
collected = 0
best_loss = 5000
semaphore = 0
count = 0

weights = model.get_weights()

channel = grpc.insecure_channel('localhost:8888',
                                    options = [('grpc.max_send_message_length', 1024*1024*1024),
                                               ('grpc_max_receive_message_length', 1024*1024*1024)])
stub = grpc_pb2_grpc.ValidatorStub(channel)


class Updater(grpc_pb2_grpc.UpdaterServicer):
    def __init__(self):
        global collected
        global weights
        global model
        global count
    def sendModel(self, request, contest):
        print("received model")
        model = Unet.Unet()
        weights = model.get_weights()

        local_model = pickle.loads(request.model)
        local_model = model_from_json(local_model)

        whole_size = request.whole_size
        local_size = request.batch_size

        collected += local_size

        modifyModel(local_model, whole_size, local_size)

        while whole_size != collected:
            pass
        model.set_weights(weights)

        val_result = checkLoss(model)

        if val_result:
            json = model.to_json()
            json = pickle.dumps(json)
            reply = grpc_pb2.updateReply(model = json, train = True)
        else:
            json = [0]
            json = pickle.dumps(json)
            if count >= 10: reply = grpc_pb2.updateReply(model = json, train = False)
            else: reply = grpc_pb2.updateReply(model = json, train = True)
        return reply



def modifyModel(local_model, whole_size, local_size):
    global weights
    local_weight = local_model.get_weights()
    for i, l in enumerate(local_weight):
        weights[i] += (l*(local_size/whole_size))

def checkLoss():
    global stub
    global count
    global best_loss

    val_model = Unet.Unet()
    val_model.set_weights(weights)

    json = pickle.dumps(val_model.to_json())
    loss = stub.validation(grpc_pb2.valRequest(model = json))

    if best_loss > loss:
        best_loss = loss
        return True
    else:
        count += 1
        return False


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=[
          ('grpc.max_send_message_length', 1024 * 1024 * 1024),
          ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
      ])
    grpc_pb2_grpc.add_UpdaterServicer_to_server(Updater(), server)
    server.add_insecure_port("[::]:8888")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()