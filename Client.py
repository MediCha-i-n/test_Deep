from tensorflow.keras.models import model_from_json, clone_model
from tensorflow.keras.optimizers import Adam
import Unet
import grpc
import grpc_pb2_grpc
import grpc_pb2
import pickle
from Polyp_gen import Generator


def fitting(stub, generator, model):
    best_loss = 50000
    while(1):
        prev_model = clone_model(model)
        print("learning...")

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss = 'binary_crossentropy')
        model.fit(generator.train_gen(), epochs = 1, steps_per_epoch=round(generator.length/8))

        print("evaluating...")
        loss = model.evaluate(generator.val_gen(), steps = len(generator.val_files))

        if loss > best_loss:
            print("loss does not improved")
            model.load_weights(prev_model)
        else:
            print("loss improved")
            best_loss = loss
        print("sending Model...")
        json = pickle.dumps(model.to_json())
        n_model = stub.updateModel(grpc_pb2.updateRequest(model = json))
        print("Received Model...")

        n_model = pickle.loads(n_model.model)
        model = model_from_json(n_model)
def validation():
    pass

def sendToServer():
    pass


def main():
    channel = grpc.insecure_channel('localhost:8888',
                                    options = [('grpc.max_send_message_length', 1024*1024*1024),
                                               ('grpc_max_receive_message_length', 1024*1024*1024)])
    stub = grpc_pb2_grpc.UpdaterStub(channel)
    generator = Generator(patch_size=256, batch_size=8)

    model = Unet.Unet()

    train(stub, generator, model)




if __name__ == "__main__":
    main()