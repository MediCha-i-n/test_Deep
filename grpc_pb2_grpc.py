# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import grpc_pb2 as grpc__pb2


class UpdaterStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.sendModel = channel.unary_unary(
                '/grpc.Updater/sendModel',
                request_serializer=grpc__pb2.updateRequest.SerializeToString,
                response_deserializer=grpc__pb2.updateReply.FromString,
                )


class UpdaterServicer(object):
    """Missing associated documentation comment in .proto file."""

    def sendModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_UpdaterServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'sendModel': grpc.unary_unary_rpc_method_handler(
                    servicer.sendModel,
                    request_deserializer=grpc__pb2.updateRequest.FromString,
                    response_serializer=grpc__pb2.updateReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'grpc.Updater', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Updater(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def sendModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/grpc.Updater/sendModel',
            grpc__pb2.updateRequest.SerializeToString,
            grpc__pb2.updateReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class ValidatorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.validation = channel.unary_unary(
                '/grpc.Validator/validation',
                request_serializer=grpc__pb2.valRequest.SerializeToString,
                response_deserializer=grpc__pb2.valReply.FromString,
                )


class ValidatorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def validation(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ValidatorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'validation': grpc.unary_unary_rpc_method_handler(
                    servicer.validation,
                    request_deserializer=grpc__pb2.valRequest.FromString,
                    response_serializer=grpc__pb2.valReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'grpc.Validator', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Validator(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def validation(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/grpc.Validator/validation',
            grpc__pb2.valRequest.SerializeToString,
            grpc__pb2.valReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
