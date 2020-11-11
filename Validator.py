from tensorflow.keras.models import model_from_json, clone_model
from tensorflow.keras.optimizers import Adam
import Unet
import grpc
import grpc_pb2_grpc
import grpc_pb2
import pickle
from concurrent import futures
from Polyp_gen import train_Generator
import tensorflow as tf

class Validator(grpc_pb2_grpc.ValidatorServicer):
    def validation(self, request, context):
        data = [0,0,0,0,0,0]
        '''
        validation data 받는 부분
        data = 
        '''
        model = pickle.loads(request.model)
        loss = model.evaluate(data)
        return loss

if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=[
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ])
    grpc_pb2_grpc.add_ValidatorServicer_to_server(Validator(), server)
    server.add_insecure_port("[::]:8888")
    server.start()
