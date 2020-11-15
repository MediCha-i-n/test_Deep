from tensorflow.keras.models import model_from_json, clone_model
from tensorflow.keras.optimizers import Adam
import Unet
import grpc
import grpc_pb2_grpc
import grpc_pb2
import pickle
from concurrent import futures
from Polyp_gen import Generator
import tensorflow as tf
import argparse


def fitting(stub, generator, model):
    train = True
    while train:
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss = 'binary_crossentropy')
        model.fit(generator.generator(), epochs = 1, steps_per_epoch=round(generator.patch_length/8))
        model, train = sendToServer(stub, model, generator)


def sendToServer(stub, model, generator):
    json = pickle.dumps(model.to_json())
    reply = stub.sendModel(grpc_pb2.updateRequest(model = json, batch_size = generator.length, whole_size = generator.whole_size, patch_size= generator.patch_length))
    n_model = pickle.loads(reply.model)
    result = reply.train

    if n_model == [0]:
        return model, result
    else:
        return model_from_json(n_model), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('identity', help = 'identity', type = str)
    args = parser.parse_args()
    model = Unet.Unet()
    if not (args.identity):
        parser.error("no identity")


    channel = grpc.insecure_channel('localhost:8888',
                                    options = [('grpc.max_send_message_length', 1024*1024*1024),
                                               ('grpc_max_receive_message_length', 1024*1024*1024)])
    stub = grpc_pb2_grpc.UpdaterStub(channel)
    generator = Generator(patch_size=256, batch_size=8, identity=args.identity)




    fitting(stub, generator, model)




if __name__ == "__main__":
    main()
