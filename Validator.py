from tensorflow.keras.models import model_from_json, clone_model
from tensorflow.keras.optimizers import Adam
import Unet
import grpc
import grpc_pb2_grpc
import grpc_pb2
import pickle
from concurrent import futures
from Polyp_gen import Generator
import Unet
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class Validator(grpc_pb2_grpc.ValidatorServicer):
    def validation(self, request, context):

        generator = Generator(patch_size=256, batch_size=8, identity='validator')

        model = pickle.loads(request.model)
        model = model_from_json(model)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        loss = model.evaluate(generator.generator(), steps = round(generator.patch_length/8))
        reply = grpc_pb2.valReply(loss = loss)
        return reply

if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=[
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ])
    grpc_pb2_grpc.add_ValidatorServicer_to_server(Validator(), server)
    server.add_insecure_port("localhost:7777")
    server.start()
    server.wait_for_termination()
