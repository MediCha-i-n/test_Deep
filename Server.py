from concurrent import futures
import grpc
import grpc_pb2
import grpc_pb2_grpc
import Unet
import pickle
import numpy as np
from tensorflow.keras.models import model_from_json, clone_model
import os
import webserver_pb2_grpc
import webserver_pb2
import threading
import requests

os.environ["SET_VISIBLE_DEVICES"] = ''

model = Unet.Unet()
#python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. grpc.proto

whole_size = 0
collected = 0
best_loss = 5000
semaphore = 0
count = 0
flag= 0
val_result = False
all_patch = 0
send_flag = 0

weights = model.get_weights()

channel = grpc.insecure_channel('localhost:7777',
                                    options = [('grpc.max_send_message_length', 1024*1024*1024),
                                               ('grpc_max_receive_message_length', 1024*1024*1024)])
stub = grpc_pb2_grpc.ValidatorStub(channel)

sem = threading.Semaphore(1)

url = url = "http://61.79.117.52:5000/dl/update_weight/"
headers = {"Content-Type": "application/json"}

class Updater(grpc_pb2_grpc.UpdaterServicer):
    def sendModel(self, request, contest):
        global collected
        global weights
        global model
        global count
        global flag
        global val_result
        global all_patch
        global send_flag

        flag = 0

        print("received model")

        local_model = pickle.loads(request.model)
        local_model = model_from_json(local_model)

        whole_size = request.whole_size
        local_size = request.batch_size
        patch_size = request.patch_size

        collected += local_size
        all_patch += patch_size



        while whole_size != collected:
            pass
        modifyModel(local_model, all_patch, patch_size)

        sem.acquire()

        if flag == 0:
            model.set_weights(weights)
            val_result = checkLoss()
        flag = 1
        sem.release()

        if val_result:
            json = model.to_json()
            json = pickle.dumps(json)
            reply = grpc_pb2.updateReply(model = json, train = True)
        else:
            json = [0]
            json = pickle.dumps(json)
            if count >= 10:
                reply = grpc_pb2.updateReply(model = json, train = False)
                sem.acquire()
                if send_flag == 0:
                    json_data = model_from_json(model)
                    response = requests.post(url = url, headers = headers, json = json_data)
                send_flag = 1
                sem.release()


            else: reply = grpc_pb2.updateReply(model = json, train = True)

        model = Unet.Unet()
        weights = model.get_weights()
        collected = 0
        all_patch = 0
        send_flag = 0
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

    val_model = clone_model(model)

    json = pickle.dumps(val_model.to_json())
    loss = stub.validation(grpc_pb2.valRequest(model = json))
    loss = loss.loss
    if best_loss > loss:
        val_model.save("best_model.hdf5")
        print("loss is imporoved %s to %s" % (best_loss, loss))
        best_loss = loss
        count = 0
        return True
    elif best_loss <= loss:
        print("loss is not improved")
        count += 1
        return False


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=3), options=[
          ('grpc.max_send_message_length', 1024 * 1024 * 1024),
          ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
      ])
    grpc_pb2_grpc.add_UpdaterServicer_to_server(Updater(), server)
    server.add_insecure_port("[::]:8888")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
