import grpc
import webserver_pb2
import webserver_pb2_grpc
from concurrent import futures
import pickle
from tensorflow.keras.models import model_from_json

class Receiver(webserver_pb2_grpc.ReceiverServicer):
    def receiveModel(self, request, context):
        model = pickle.loads(request.model)
        model = model_from_json(model)

        model.save('best_model.hdf5')

        return



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options={
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    })

    webserver_pb2_grpc.add_ReceiverServicer_to_server(Receiver(), server)
    server.add_insecure_port("[::]:7777")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()