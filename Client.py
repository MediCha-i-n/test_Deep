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


def fitting(stub, generator, model):
    train = True
    while train:
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss = 'binary_crossentropy')
        model.fit(generator.train_gen(), epochs = 1, steps_per_epoch=round(generator.length/8))

        model, train = sendToServer(stub, model, generator)


def sendToServer(stub, model, generator):
    json = pickle.dumps(model.to_json())
    reply = stub.sendModel(grpc_pb2.updateRequest(model = json, batch_size = generator.train_len, whole_size = generator.whole_size))
    n_model = pickle.loads(reply.model)
    result = reply.train

    if n_model == [0]:
        return model, result
    else:
        return model_from_json(n_model), result


def main():
    channel = grpc.insecure_channel('localhost:8888',
                                    options = [('grpc.max_send_message_length', 1024*1024*1024),
                                               ('grpc_max_receive_message_length', 1024*1024*1024)])
    stub = grpc_pb2_grpc.UpdaterStub(channel)
    generator = train_Generator(patch_size=256, batch_size=8, path = "./../polyps/Kvasir-SEG")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=[
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ])
    grpc_pb2_grpc.add_ValidatorServicer_to_server(Validator(), server)
    server.add_insecure_port("[::]:8888")
    server.start()

    model = Unet.Unet()

    fitting(stub, generator, model)




if __name__ == "__main__":
    main()