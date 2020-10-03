from concurrent import futures
import grpc
import tensorflow_federated as tff
import tensorflow as tf
import grpc_pb2
import grpc_pb2_grpc
import tensorflow_federated as tff
import Unet
import pickle

model = Unet.Unet()
#python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. grpc.proto


class Updater(grpc_pb2_grpc.UpdaterServicer):
    def updateModel(self, request, context):
        print("received model")
        print("averaging model")

        n_model = pickle.loads(request.model)

        print("responsing...")

        return request


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