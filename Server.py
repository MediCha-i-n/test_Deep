from concurrent import futures
import grpc
import grpc_pb2
import grpc_pb2_grpc
import Unet
import pickle
import numpy as np

model = Unet.Unet()
#python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. grpc.proto

whole_size = 0
collected = 0

weights = model.get_weights()


class Updater(grpc_pb2_grpc.UpdaterServicer):
    def __init__(self):
        global collected
        global weights
        global model

    def updateModel(self, request, context):
        print("received model")
        print("averaging model")

        self.models.append(pickle.loads(request.model))
        self.n.append(request.size())
        self.collect += 1

        while self.collect != self.n_client:
            continue
        self.collect = 0
        new_model = np.zeros

        print("responsing...")

        return request
    def sendModel(self, request, contest):
        print("received model")
        model = Unet.Unet()
        weights = model.get_weights()

        local_model = pickle.loads(request.model)

        whole_size = request.whole_size
        local_size = request.batch_size

        collected += local_size

        modifyModel(local_model, whole_size, local_size)

        while whole_size != collected:
            pass
        model.set_weights(weights)

        checkLoss()

        return model



def modifyModel(local_model, whole_size, local_size):
    global weights
    local_weight = local_model.get_weights()
    for i, l in enumerate(local_weight):
        weights[i] += (l*(local_size/whole_size))

def checkLoss():
    pass




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