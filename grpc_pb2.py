# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: grpc.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='grpc.proto',
  package='grpc',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\ngrpc.proto\x12\x04grpc\"Z\n\rupdateRequest\x12\r\n\x05model\x18\x01 \x01(\x0c\x12\x12\n\nwhole_size\x18\x02 \x01(\x04\x12\x12\n\nbatch_size\x18\x03 \x01(\x04\x12\x12\n\npatch_size\x18\x04 \x01(\x04\"\x1b\n\nvalRequest\x12\r\n\x05model\x18\x01 \x01(\x0c\"\x18\n\x08valReply\x12\x0c\n\x04loss\x18\x01 \x01(\x02\"+\n\x0bupdateReply\x12\r\n\x05model\x18\x01 \x01(\x0c\x12\r\n\x05train\x18\x02 \x01(\x08\x32@\n\x07Updater\x12\x35\n\tsendModel\x12\x13.grpc.updateRequest\x1a\x11.grpc.updateReply\"\x00\x32=\n\tValidator\x12\x30\n\nvalidation\x12\x10.grpc.valRequest\x1a\x0e.grpc.valReply\"\x00\x62\x06proto3'
)




_UPDATEREQUEST = _descriptor.Descriptor(
  name='updateRequest',
  full_name='grpc.updateRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='grpc.updateRequest.model', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='whole_size', full_name='grpc.updateRequest.whole_size', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='grpc.updateRequest.batch_size', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='patch_size', full_name='grpc.updateRequest.patch_size', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=20,
  serialized_end=110,
)


_VALREQUEST = _descriptor.Descriptor(
  name='valRequest',
  full_name='grpc.valRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='grpc.valRequest.model', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=112,
  serialized_end=139,
)


_VALREPLY = _descriptor.Descriptor(
  name='valReply',
  full_name='grpc.valReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='loss', full_name='grpc.valReply.loss', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=141,
  serialized_end=165,
)


_UPDATEREPLY = _descriptor.Descriptor(
  name='updateReply',
  full_name='grpc.updateReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='grpc.updateReply.model', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='train', full_name='grpc.updateReply.train', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=167,
  serialized_end=210,
)

DESCRIPTOR.message_types_by_name['updateRequest'] = _UPDATEREQUEST
DESCRIPTOR.message_types_by_name['valRequest'] = _VALREQUEST
DESCRIPTOR.message_types_by_name['valReply'] = _VALREPLY
DESCRIPTOR.message_types_by_name['updateReply'] = _UPDATEREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

updateRequest = _reflection.GeneratedProtocolMessageType('updateRequest', (_message.Message,), {
  'DESCRIPTOR' : _UPDATEREQUEST,
  '__module__' : 'grpc_pb2'
  # @@protoc_insertion_point(class_scope:grpc.updateRequest)
  })
_sym_db.RegisterMessage(updateRequest)

valRequest = _reflection.GeneratedProtocolMessageType('valRequest', (_message.Message,), {
  'DESCRIPTOR' : _VALREQUEST,
  '__module__' : 'grpc_pb2'
  # @@protoc_insertion_point(class_scope:grpc.valRequest)
  })
_sym_db.RegisterMessage(valRequest)

valReply = _reflection.GeneratedProtocolMessageType('valReply', (_message.Message,), {
  'DESCRIPTOR' : _VALREPLY,
  '__module__' : 'grpc_pb2'
  # @@protoc_insertion_point(class_scope:grpc.valReply)
  })
_sym_db.RegisterMessage(valReply)

updateReply = _reflection.GeneratedProtocolMessageType('updateReply', (_message.Message,), {
  'DESCRIPTOR' : _UPDATEREPLY,
  '__module__' : 'grpc_pb2'
  # @@protoc_insertion_point(class_scope:grpc.updateReply)
  })
_sym_db.RegisterMessage(updateReply)



_UPDATER = _descriptor.ServiceDescriptor(
  name='Updater',
  full_name='grpc.Updater',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=212,
  serialized_end=276,
  methods=[
  _descriptor.MethodDescriptor(
    name='sendModel',
    full_name='grpc.Updater.sendModel',
    index=0,
    containing_service=None,
    input_type=_UPDATEREQUEST,
    output_type=_UPDATEREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_UPDATER)

DESCRIPTOR.services_by_name['Updater'] = _UPDATER


_VALIDATOR = _descriptor.ServiceDescriptor(
  name='Validator',
  full_name='grpc.Validator',
  file=DESCRIPTOR,
  index=1,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=278,
  serialized_end=339,
  methods=[
  _descriptor.MethodDescriptor(
    name='validation',
    full_name='grpc.Validator.validation',
    index=0,
    containing_service=None,
    input_type=_VALREQUEST,
    output_type=_VALREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_VALIDATOR)

DESCRIPTOR.services_by_name['Validator'] = _VALIDATOR

# @@protoc_insertion_point(module_scope)
